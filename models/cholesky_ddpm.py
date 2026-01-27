import torch
import torch.nn as nn
from einops import rearrange, repeat
from utils.config_utils import *
import torch.nn.functional as F

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb
    
class ResBlock(nn.Module):
    r"""
    A Residual MLP Block that handles:
    1. Time Embeddings (Scale & Shift)
    2. Cross Attention (Optional)
    3. Residual Dense Layers
    """
    def __init__(self, in_dim=256, out_dim=256, t_emb_dim=256, num_heads=8, num_layers=4, num_vae_comp=16,
                 cross_attn=False, context_dim=None, dropout=0.1):
        super().__init__()
        self.cross_attn = cross_attn
        self.out_dim = out_dim

        # 1. Input Projection / Norm
        self.norm1 = nn.LayerNorm(in_dim)
        self.proj_in = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        # 2. Time Embedding Projection (Shift and Scale)
        # We map time embedding to out_dim * 2 to do (gamma * x) + beta
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_dim * 2 * num_vae_comp)
        )

        # 3. Feed Forward MLP
        self.net = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # 4. Cross Attention (if text/context conditioned)
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.attn_norm = nn.ModuleList(
                [nn.LayerNorm(out_dim)
                 for _ in range(num_layers)]
            )
            self.attn = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_dim)
                 for _ in range(num_layers)]
            )
        self.num_layers = num_layers
        # Residual connection adjustment
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, t_emb, context=None):
        # x: [B, N_component, in_dim], default: [B, 16, 256]
        B, ncomponent, in_dim = x.shape
        # 1. Initial Residual Path
        h = self.proj_in(self.norm1(x)) # [B, N, out_dim]
        
        # 2. Add Time Embedding (Adaptive Layer Norm style)
        # t_emb: [B, t_dim] -> [B, out_dim * 2 * ncomponent]
        t_chunk = self.t_proj(t_emb)
        t_scale, t_shift = t_chunk.chunk(2, dim=1)
        t_scale = torch.stack(t_scale.chunk(ncomponent, dim=1), 1)
        t_shift = torch.stack(t_shift.chunk(ncomponent, dim=1), 1)
        
        # Broadcast t over the sequence dimension N
        h = h * (1 + t_scale) + t_shift
        
        # 3. MLP Processing
        h = self.net(h)
        
        # 4. Cross Attention
        if self.cross_attn:
            assert context is not None
            # Query = h, Key/Value = context
            for li in range(self.num_layers):
                norm_h = self.attn_norm[li](h)
                ctx_proj = self.context_proj[li](context) # Align context dim
                
                attn_out, _ = self.attn[li](query=norm_h, key=ctx_proj, value=ctx_proj)
                h = h + attn_out

        # Final Residual
        return h + self.res_proj(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, t_emb_dim, num_vae_comp, num_heads=8, config=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        curr_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(ResBlock(
                in_dim=curr_dim, 
                out_dim=h_dim, 
                t_emb_dim=t_emb_dim,
                cross_attn=config.get('text_cond', False),
                context_dim=config.get('text_embed_dim', None),
                num_vae_comp=num_vae_comp,
                num_heads=num_heads,
            ))
            curr_dim = h_dim
        
        self.out_dim = curr_dim

    def forward(self, x, t_emb, context=None):
        skips = []
        for layer in self.layers:
            x = layer(x, t_emb, context)
            skips.append(x)
        return x, skips

def to_correlation_matrix(raw_output):
    """
    Projects raw output to a Correlation Matrix (SPD with diagonal = 1).
    Steps:
    1. Interpret raw output as Lower Triangular L.
    2. Compute Covariance V = L @ L.T
    3. Normalize V to Correlation C: C_ij = V_ij / sqrt(V_ii * V_jj)
    """
    B, N = raw_output.shape[:2]
    
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
    
    # Add channel dim back: (B, N, N)
    return C
    
class CholeskyComp(nn.Module):
    def __init__(self, in_dim, out_dim, matrix_size=116):
        super().__init__()
        self.matrix_size = matrix_size
        self.net = nn.Linear(in_dim, out_dim)

        
    def forward(self, x):
        return to_correlation_matrix(self.net(x))
        

class MLPCholeskyDecoder(nn.Module):
    r"""
    Decoder path. Named 'CholeskyDecoder' per request, usually acts as 
    the denoising reconstruction head in DDPM.
    """
    def __init__(self, out_dim, hidden_dims, t_emb_dim, num_vae_comp, num_heads, config):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.matrix_size = 116
        
        # Hidden dims are reversed for decoder
        curr_dim = hidden_dims[0] 
        
        # Skip connection handling: Decoder input is concat(latent, skip)
        # So input dim is usually doubled if simple concat, or same if summed. 
        # Here we assume simple residual flow + skips logic handled in main.

        if out_dim != self.matrix_size:
            layer_dims = hidden_dims[1:] + [out_dim]
        else:
            layer_dims = hidden_dims[1:] 
            
        for i, h_dim in enumerate(layer_dims):
            # If we concat skips, input is curr_dim * 2, else curr_dim
            # Let's assume we sum skips for simpler MLP logic or project them.
            self.layers.append(ResBlock(
                in_dim=curr_dim, # We will handle skip combination in forward
                out_dim=h_dim, 
                t_emb_dim=t_emb_dim,
                cross_attn=config.get('text_cond', False),
                context_dim=config.get('text_embed_dim', None),
                num_vae_comp=num_vae_comp,
                num_heads=num_heads if out_dim != self.matrix_size and h_dim==out_dim else 8,
            ))
            curr_dim = h_dim
        
        if out_dim == self.matrix_size:
            self.layers.append(CholeskyComp(layer_dims[-1], out_dim, matrix_size=self.matrix_size))
        
    def forward(self, x, skips, t_emb, context=None):
        for layer in self.layers:
            # Pop skip connection from encoder
            skip = skips.pop()
            
            # Simple skip strategy: Additive (requires dimensions to match)
            # If dims don't match, we assume the architecture is symmetric
            if skip.shape[-1] == x.shape[-1]:
                x = x + skip 
            if not isinstance(layer, CholeskyComp):
                x = layer(x, t_emb, context)
            else:
                x = layer(x)
        
        # return self.to_correlation_matrix(x)
        return x


class LatentMLPDiffusion(nn.Module):
    def __init__(self, model_config, num_heads=8, input_shape=[16, 256]):
        super().__init__()
        # input_shape: tuple (N_component, N_codebookdim)
        self.n_component, self.n_dim = input_shape
        if self.n_component == self.n_dim: 
            self.n_component = 1
            self.n_dim = len(torch.tril_indices(self.n_dim, self.n_dim, -1)[0])
            # print(self.n_dim)
        
        # Config Extraction
        # self.hidden_layers = model_config.get('hidden_layers', [256, 512, 512]) # Example MLP depth
        self.hidden_layers = model_config.get('hidden_layers', [2048, 1024, 512]) # 
        self.t_emb_dim = model_config['time_emb_dim']
        
        # Conditioning Flags
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        self.text_cond = False
        self.class_cond = False
        self.text_embed_dim = None
        
        # Parse Conditioning (Same logic as your U-Net)
        if self.condition_config:
            c_types = self.condition_config.get('condition_types', [])
            if 'text' in c_types:
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'class' in c_types:
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
                self.class_emb = nn.Embedding(self.num_classes, self.t_emb_dim)

        # Config dict for sub-modules
        sub_config = {'text_cond': self.text_cond, 'text_embed_dim': self.text_embed_dim}

        # 1. Time Projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # 2. MLP Encoder
        self.encoder = MLPEncoder(
            input_dim=self.n_dim, 
            hidden_dims=self.hidden_layers, 
            t_emb_dim=self.t_emb_dim, 
            num_vae_comp=self.n_component,
            config=sub_config
        )

        # 3. Middle Block (Bottleneck)
        self.mid_block = ResBlock(
            in_dim=self.hidden_layers[-1],
            out_dim=self.hidden_layers[-1],
            t_emb_dim=self.t_emb_dim,
            cross_attn=self.text_cond,
            context_dim=self.text_embed_dim,
            num_vae_comp=self.n_component
        )

        # 4. Cholesky Decoder
        # Reverse hidden layers for decoder path
        rev_hidden = list(reversed(self.hidden_layers))
        self.decoder = MLPCholeskyDecoder(
            out_dim=self.n_dim, # Project back to original feature dim
            hidden_dims=rev_hidden,
            t_emb_dim=self.t_emb_dim,
            num_vae_comp=self.n_component,
            num_heads=num_heads, config=sub_config
        )

    def forward(self, x, t, cond_input=None):
        # x shape: [B, N_component, N_dim]
        # t shape: [B]
        
        # 1. Time Embedding
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb) # [B, t_dim]

        # 2. Class Conditioning
        if self.class_cond:
            assert cond_input is not None
            # validate_class_conditional_input(cond_input, x, self.num_classes)
            # Add class embedding to time embedding
            c_emb = self.class_emb(cond_input['class'].long()) # [B, t_dim]
            t_emb = t_emb + c_emb

        # 3. Text Conditioning
        context = None
        if self.text_cond:
            assert cond_input is not None and 'text' in cond_input
            context = cond_input['text'] # [B, Seq, Ctx_Dim]

        # 4. Forward Pass
        # Encoder
        h, skips = self.encoder(x, t_emb, context)
        
        # Bottleneck
        h = self.mid_block(h, t_emb, context)
        
        # Decoder (Cholesky Decoder)
        out = self.decoder(h, skips, t_emb, context)
        
        return out