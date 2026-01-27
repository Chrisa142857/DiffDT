import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock

import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.codebook_size = model_config['codebook_size']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################
        
        ##################### Decoder ######################
        
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i-1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()
        
        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
    
    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses


class SPD_VQVAE(nn.Module):
    def __init__(self, model_config, input_dim=116, **kwargs):
        """
        input_dim: The size of the matrix N (for an N x N matrix). 
                   The model expects input shape (Batch, N, N) or (Batch, 1, N, N).
        """
        super().__init__()
        
        self.matrix_size = input_dim
        self.flat_input_size = input_dim * input_dim
        
        # Configuration
        self.enc_hidden_sizes = model_config.get('enc_hidden_sizes', [512, 256, 128])
        self.dec_hidden_sizes = model_config.get('dec_hidden_sizes', [128, 256, 512])
        
        # How many codebook indices to represent one matrix
        self.num_latents = model_config.get('num_latents', 16) if 'num_latents' not in kwargs else kwargs['num_latents']
        # Dimension of each code
        self.z_dim = model_config.get('z_dim', 256) if 'z_dim' not in kwargs else kwargs['z_dim']
        self.codebook_size = model_config.get('codebook_size', 256) if 'codebook_size' not in kwargs else kwargs['codebook_size']
        
        # --- ENCODER (MLP) ---
        enc_layers = []
        in_dim = self.flat_input_size
        
        for h_dim in self.enc_hidden_sizes:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            enc_layers.append(nn.BatchNorm1d(h_dim)) # BN helps MLPs train faster
            enc_layers.append(nn.SiLU())
            in_dim = h_dim
            
        self.encoder_body = nn.Sequential(*enc_layers)
        
        # Final projection to latent space (Batch, Num_Latents * Z_Dim)
        self.encoder_out = nn.Linear(in_dim, self.num_latents * self.z_dim)
        
        # --- CODEBOOK ---
        self.embedding = nn.Embedding(self.codebook_size, self.z_dim)
        
        # --- DECODER (MLP) ---
        # Input is (Batch, Num_Latents * Z_Dim)
        dec_layers = []
        in_dim = self.num_latents * self.z_dim
        
        for h_dim in self.dec_hidden_sizes:
            dec_layers.append(nn.Linear(in_dim, h_dim))
            dec_layers.append(nn.BatchNorm1d(h_dim))
            dec_layers.append(nn.SiLU())
            in_dim = h_dim
            
        self.decoder_body = nn.Sequential(*dec_layers)
        
        # Final projection back to flattened matrix size
        self.decoder_out = nn.Linear(in_dim, self.flat_input_size)

    def quantize(self, z):
        # z shape: (Batch, Num_Latents, Z_Dim)
        B, T, C = z.shape
        
        # Flatten to (Batch * Num_Latents, Z_Dim) for distance calc
        flat_z = z.view(-1, C)
        
        # Calculate distances to codebook vectors
        # (B*T, C) vs (K, C) -> (B*T, K)
        dists = torch.cdist(flat_z, self.embedding.weight)
        
        # Find nearest indices
        min_encoding_indices = torch.argmin(dists, dim=1)
        
        # Get quantized vectors
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices)
        
        # Reshape back to (B, T, C)
        quant_out = quant_out.view(B, T, C)
        
        # Losses
        commitment_loss = torch.mean((quant_out.detach() - z) ** 2)
        codebook_loss = torch.mean((quant_out - z.detach()) ** 2)
        quant_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }
        
        # Straight-through estimator (gradients flow through z)
        quant_out = z + (quant_out - z).detach()
        
        return quant_out, quant_losses, min_encoding_indices

    def encode(self, x):
        # x: (Batch, 1, N, N) or (Batch, N, N)
        B = x.shape[0]
        
        # Flatten: (B, N*N)
        x_flat = x.view(B, -1)
        
        out = self.encoder_body(x_flat)
        out = self.encoder_out(out)
        
        # Reshape for quantization: (Batch, Num_Latents, Z_Dim)
        out = out.view(B, self.num_latents, self.z_dim)
        
        # Quantize
        z_q, losses, indices = self.quantize(out)
        
        return z_q, losses, indices

    def to_correlation_matrix(self, raw_output):
        """
        Projects raw output to a Correlation Matrix (SPD with diagonal = 1).
        Steps:
        1. Interpret raw output as Lower Triangular L.
        2. Compute Covariance V = L @ L.T
        3. Normalize V to Correlation C: C_ij = V_ij / sqrt(V_ii * V_jj)
        """
        B = raw_output.shape[0]
        N = self.matrix_size
        
        # Reshape to Matrix: (B, N, N)
        L = raw_output.view(B, N, N)
        
        # 1. Lower Triangular Mask
        L = torch.tril(L)
        
        # 2. Positive Diagonal (Cholesky requirement)
        # Create diagonal mask
        diag_mask = torch.eye(N, device=L.device).bool().unsqueeze(0).expand(B, -1, -1)
        # Softplus to ensure > 0
        L = torch.where(diag_mask, F.softplus(L) + 1e-6, L)
        
        # 3. Compute Covariance Matrix (SPD)
        # V = L * L^T
        V = torch.bmm(L, L.transpose(1, 2))
        
        # 4. Normalize to Correlation Matrix (Diagonal = 1)
        # Extract the diagonal: (B, N)
        v_diag = torch.diagonal(V, dim1=-2, dim2=-1)
        
        # Compute inverse sqrt of diagonal: (B, N)
        # 1 / sqrt(sigma^2)
        std_inv = 1.0 / torch.sqrt(v_diag + 1e-8)
        
        # Create outer product for normalization: (B, N, N)
        # Matrix M where M_ij = std_inv_i * std_inv_j
        norm_matrix = torch.bmm(std_inv.unsqueeze(2), std_inv.unsqueeze(1))
        
        # Apply normalization: C = V * M
        C = V * norm_matrix
        
        # Add channel dim back: (B, 1, N, N)
        return C.unsqueeze(1), L

    def decode(self, z_q):
        # z_q: (Batch, Num_Latents, Z_Dim)
        B = z_q.shape[0]
        
        # Flatten for MLP: (Batch, Num_Latents * Z_Dim)
        z_flat = z_q.view(B, -1)
        
        out = self.decoder_body(z_flat)
        
        # Raw projection (Batch, N*N)
        raw_out = self.decoder_out(out)
        
        # GEOMETRIC PROJECTION
        spd_out, L = self.to_correlation_matrix(raw_out)
        
        return spd_out, L

    def forward(self, x):
        z_q, losses, _ = self.encode(x)
        out, L = self.decode(z_q)
        return out, z_q, losses, L