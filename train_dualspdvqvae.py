import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

# --- Custom Model Imports ---
from models.vqvae import VQVAE, SPD_VQVAE
from models.lpips import LPIPS
# from models.discriminator import Discriminator
from train_discriminator import Discriminator

# --- Dataset Imports ---
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.icdfc_dataset import ICDFCDataset, FCICDDataset
from transformers import get_scheduler
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:1'

def decompose_fc(batch_mat, threshold=50, dec_bg=True):
    tril_idx = torch.tril_indices(batch_mat.shape[-1], batch_mat.shape[-1])
    triu_idx = torch.triu_indices(batch_mat.shape[-1], batch_mat.shape[-1])
    batch = batch_mat[:, tril_idx[0], tril_idx[1]].clone()
    fourier_coefficients = torch.fft.rfft(batch, dim=-1)
    magnitude_spectrum = torch.abs(fourier_coefficients)
    if dec_bg :
        cond = magnitude_spectrum > threshold 
    else:
        cond = magnitude_spectrum <= threshold 
    fourier_coefficients[cond] = 0
    decomposed_signal = torch.fft.irfft(fourier_coefficients, n=batch.shape[-1], dim=-1)
    batch_mat = torch.zeros_like(batch_mat)
    batch_mat[:, tril_idx[0], tril_idx[1]] = decomposed_signal
    batch_mat[:, triu_idx[0], triu_idx[1]] = batch_mat[:, triu_idx[1], triu_idx[0]]
    return batch_mat

def reverse_decomp(comp1, comp2):
    n = comp1.shape[-1]
    batch_mat = torch.zeros(len(comp1), n, n)
    tril_idx = torch.tril_indices(n, n)
    triu_idx = torch.triu_indices(n, n)
    batch_mat[:, tril_idx[0], tril_idx[1]] = torch.fft.irfft(torch.fft.rfft(comp1[:, tril_idx[0], tril_idx[1]], dim=-1) + torch.fft.rfft(comp2[:, tril_idx[0], tril_idx[1]], dim=-1), n=tril_idx.shape[-1], dim=-1).detach().cpu()
    batch_mat[:, triu_idx[0], triu_idx[1]] = batch_mat[:, triu_idx[1], triu_idx[0]]
    return batch_mat
    
def train(args):
    device = args.device
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    savetag = f'DualSPD_thr{args.thr}{args.note}'
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    train_config['autoencoder_batch_size'] = 2048
    train_config['autoencoder_epochs'] = args.epochs
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # --- MODEL SELECTION LOGIC ---
    # We choose the MLP architecture if dealing with Matrix data (ICDFC/FCICD)
    is_matrix_data = dataset_config['name'] in ['ICDFC', 'FCICD']
    
    if is_matrix_data:
        print(f"Initializing SPD_VQVAE for matrix size {dataset_config['im_size']}...")
        model1 = SPD_VQVAE(input_dim=dataset_config['im_size'],
                           num_latents=args.m1_num_latent, # same
                              model_config=autoencoder_config).to(device)
        # LPIPS (Perceptual Loss) is designed for Natural Images (RGB). 
        # It is meaningless/harmful for Correlation Matrices.
        model2 = SPD_VQVAE(input_dim=dataset_config['im_size'],
                          model_config=autoencoder_config,
                           num_latents=args.m2_num_latent, # 
                           codebook_size=args.m2_num_latent*args.m2_codebook # 
                          ).to(device)
        use_lpips = False
    else:
        print("Initializing Standard ConvNet VQVAE...")
        model = VQVAE(im_channels=dataset_config['im_channels'],
                      model_config=autoencoder_config).to(device)
        use_lpips = True

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'ICDFC': ICDFCDataset,
        'FCICD': FCICDDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train', onlyrest=True, use_latents=False,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    val_im_dataset = im_dataset_cls(split='val', onlyrest=True, use_latents=False,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    val_data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.L1Loss()
    # Disc Loss
    # disc_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize LPIPS only if working with natural images
    if use_lpips:
        lpips_model = LPIPS(device=device).eval().to(device)
    else:
        lpips_model = None

    # discriminator = Discriminator().to(device)
    # discriminator.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                         'SPD'+train_config['vqvae_discriminator_ckpt_name']), map_location=device))
    
    # discriminator.eval()
    # for param in discriminator.parameters():
    #     param.requires_grad = False
    # optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    lr = 0.00005
    optimizer_g = Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr, betas=(0.5, 0.999))
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer_g,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    save_step_count = 0
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = 64 #train_config['autoencoder_img_save_steps']
    img_save_count = 0
    best_loss = 1e+10
    best_model1 = None
    
    for epoch_idx in range(num_epochs):
        recon_losses1 = []
        recon_losses2 = []
        codebook_losses = []
        # perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        
        model1.train()
        model2.train()
        # Wrapping loader in tqdm for progress bar
        for im, cholesky in data_loader:
            step_count += 1
            # beh = beh.squeeze()
            im = im.float().to(device)
            im1 = decompose_fc(im, dec_bg=False, threshold=args.thr)[:, None]
            im2 = decompose_fc(im, dec_bg=True, threshold=args.thr)[:, None]
            model_output = model1(im1)
            output1, z, quantize_losses1, L_pred = model_output
            model_output = model2(im2)
            output2, z, quantize_losses2, L_pred = model_output
            ######### Optimize Generator ##########
            # 1. Reconstruction Loss (MSE)
            recon_loss1 = recon_criterion(output1, im1) 
            recon_loss2 = recon_criterion(output2, im2) 
            recon_losses1.append(recon_loss1.item())
            recon_losses2.append(recon_loss2.item())
            # recon_loss = recon_loss / acc_steps
            recon_loss1 = recon_loss1 / acc_steps
            recon_loss2 = recon_loss2 / acc_steps
            # cholesky_loss = recon_criterion(L_pred, cholesky.to(device))
            
            g_loss = recon_loss1 + recon_loss2
            # g_loss = cholesky_loss / acc_steps
            
            # 2. VQ Codebook Losses
            # Note: SPD_VQVAE might return scalar or tensor, ensuring items are extracted correctly
            g_loss += (train_config['codebook_weight'] * quantize_losses1['codebook_loss'] / acc_steps)
            g_loss += (train_config['commitment_beta'] * quantize_losses1['commitment_loss'] / acc_steps)
            g_loss += (train_config['codebook_weight'] * quantize_losses2['codebook_loss'] / acc_steps)
            g_loss += (train_config['commitment_beta'] * quantize_losses2['commitment_loss'] / acc_steps)
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses1['codebook_loss'].item() + train_config['codebook_weight'] * quantize_losses2['codebook_loss'].item())
            # gen_losses.append(cholesky_loss.item())

            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                lr_scheduler.step()
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        val_recon_losses1 = []
        val_recon_losses2 = []
        val_codebook_losses = []
        val_gen_losses = []
        val_losses = []
        model1.eval()
        model2.eval()
        for im, cholesky in val_data_loader:
            save_step_count += 1
            with torch.no_grad():
                im = im.float().to(device)
                im1 = decompose_fc(im, dec_bg=False, threshold=args.thr)[:, None]
                im2 = decompose_fc(im, dec_bg=True, threshold=args.thr)[:, None]
                model_output = model1(im1)
                output1, z, quantize_losses1, L_pred = model_output
                model_output = model2(im2)
                output2, z, quantize_losses2, L_pred = model_output

            # --- Image/Matrix Saving Logic ---
            if save_step_count % image_save_steps == 0 or save_step_count == 1:
                sample_size = min(8, im.shape[0])
                
                # Reshape/Clamp for visualization
                # Correlation matrices are [-1, 1], so this normalization is correct for PNG saving
                save_output1 = torch.clamp(output1[:sample_size], -1., 1.).detach().cpu().squeeze()[:, None]
                save_output1 = ((save_output1 + 1) / 2)
                save_input1 = ((im1[:sample_size] + 1) / 2).detach().cpu()
                save_output2 = torch.clamp(output2[:sample_size], -1., 1.).detach().cpu().squeeze()[:, None]
                save_output2 = ((save_output2 + 1) / 2)
                save_input2 = ((im2[:sample_size] + 1) / 2).detach().cpu()
                save_output3 = torch.clamp(reverse_decomp(output1.squeeze(), output2.squeeze())[:sample_size], -1., 1.).detach().cpu().squeeze()[:, None]
                save_output3 = ((save_output3 + 1) / 2)
                save_input3 = ((im[:sample_size, None] + 1) / 2).detach().cpu()
                
                
                grid = make_grid(torch.cat([save_input1, save_output1, save_input2, save_output2, save_input3, save_output3], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                
                save_dir = os.path.join(train_config['task_name'], f'{savetag}vqvae_autoencoder_samples')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                img.save(os.path.join(save_dir, 'val_current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
                if best_model1 is not None:
                    torch.save(best_model1, os.path.join(train_config['task_name'],
                                                                f'{savetag}_m1'+train_config['vqvae_autoencoder_ckpt_name']))
                    torch.save(best_model2, os.path.join(train_config['task_name'],
                                                                f'{savetag}_m2'+train_config['vqvae_autoencoder_ckpt_name']))
            ######### Optimize Generator ##########
            # 1. Reconstruction Loss (MSE)
            # recon_loss = recon_criterion(output, im) 
            recon_loss1 = recon_criterion(output1, im1) 
            recon_loss2 = recon_criterion(output2, im2) 
            val_recon_losses1.append(recon_loss1.item())
            val_recon_losses2.append(recon_loss2.item())
            # val_recon_losses.append(recon_loss.item())
            recon_loss1 = recon_loss1 / acc_steps
            recon_loss2 = recon_loss2 / acc_steps
            # cholesky_loss = recon_criterion(L_pred, cholesky.to(device))
            
            g_loss = recon_loss1 + recon_loss2
            # g_loss = cholesky_loss / acc_steps
            
            # 2. VQ Codebook Losses
            # Note: SPD_VQVAE might return scalar or tensor, ensuring items are extracted correctly
            g_loss += (train_config['codebook_weight'] * quantize_losses1['codebook_loss'] / acc_steps)
            g_loss += (train_config['commitment_beta'] * quantize_losses1['commitment_loss'] / acc_steps)
            g_loss += (train_config['codebook_weight'] * quantize_losses2['codebook_loss'] / acc_steps)
            g_loss += (train_config['commitment_beta'] * quantize_losses2['commitment_loss'] / acc_steps)
            val_codebook_losses.append(train_config['codebook_weight'] * quantize_losses1['codebook_loss'].item() + train_config['codebook_weight'] * quantize_losses2['codebook_loss'].item())
            # val_gen_losses.append(cholesky_loss.item())

            val_losses.append(g_loss.item())
            # g_loss.backward()
        
        log_string = (
            'Finished epoch: {} | Train Recon Loss 1 : {:.4f} | Train Recon Loss 2 : {:.4f} | Train Codebook : {:.4f} | '
            'Train Cholesky Loss : {:.4f} | Val Recon Loss 1 : {:.4f} | Val Recon Loss 2 : {:.4f} | Val Codebook : {:.4f} | '
            'Val Cholesky Loss : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(recon_losses1),
                np.mean(recon_losses2),
                np.mean(codebook_losses),
                np.mean(gen_losses) if len(gen_losses) > 0 else 0,
                np.mean(val_recon_losses1),
                np.mean(val_recon_losses2),
                np.mean(val_codebook_losses),
                np.mean(val_gen_losses) if len(val_gen_losses) > 0 else 0,
                
            )
        )
        print(log_string)
        if np.mean(val_losses) < best_loss:
            best_model1 = {k: v.cpu() for k,v in model1.state_dict().items()}
            best_model2 = {k: v.cpu() for k,v in model2.state_dict().items()}
    torch.save(best_model1, os.path.join(train_config['task_name'],
                                                f'{savetag}_m1'+train_config['vqvae_autoencoder_ckpt_name']))
    torch.save(best_model2, os.path.join(train_config['task_name'],
                                                f'{savetag}_m2'+train_config['vqvae_autoencoder_ckpt_name']))
    
    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/icd2fc_image_cond.yaml', type=str)
    parser.add_argument('--thr', default=50, type=int)
    parser.add_argument('--m1_num_latent', default=256, type=int)
    parser.add_argument('--m2_num_latent', default=16, type=int)
    parser.add_argument('--m2_codebook', default=1, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--note', default='', type=str)
    args = parser.parse_args()
    train(args)