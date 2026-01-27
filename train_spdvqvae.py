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
device = 'cuda:1'

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    savetag = 'SPD_reconL_noreconM'
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    train_config['autoencoder_batch_size'] = 2048
    
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
        model = SPD_VQVAE(input_dim=dataset_config['im_size'],
                              model_config=autoencoder_config).to(device)
        # LPIPS (Perceptual Loss) is designed for Natural Images (RGB). 
        # It is meaningless/harmful for Correlation Matrices.
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
    
    im_dataset = im_dataset_cls(split='train', onlyrest=False, use_latents=False,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    val_im_dataset = im_dataset_cls(split='val', onlyrest=False, use_latents=False,
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
    optimizer_g = Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
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
    
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        # perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        # optimizer_d.zero_grad()
        
        # Wrapping loader in tqdm for progress bar
        for im, cholesky in data_loader:
            step_count += 1
            # beh = beh.squeeze()
            im = im.float().to(device)[:, None]
            
            # Fetch autoencoders output (reconstructions)
            # Both models return: (reconstructed_x, quantized_z, losses_dict)
            model_output = model(im)
            output, z, quantize_losses, L_pred = model_output
            # --- Image/Matrix Saving Logic ---
            # if step_count % image_save_steps == 0 or step_count == 1:
            #     sample_size = min(8, im.shape[0])
                
            #     # Reshape/Clamp for visualization
            #     # Correlation matrices are [-1, 1], so this normalization is correct for PNG saving
            #     save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
            #     save_output = ((save_output + 1) / 2)
                
            #     save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
            #     grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
            #     img = torchvision.transforms.ToPILImage()(grid)
                
            #     save_dir = os.path.join(train_config['task_name'], f'{savetag}vqvae_autoencoder_samples')
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     img.save(os.path.join(save_dir, 'current_autoencoder_sample_{}.png'.format(img_save_count)))
            #     img_save_count += 1
            #     img.close()
            
            ######### Optimize Generator ##########
            # 1. Reconstruction Loss (MSE)
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            cholesky_loss = recon_criterion(L_pred, cholesky.to(device))
            
            # g_loss = recon_loss + cholesky_loss / acc_steps
            g_loss = cholesky_loss / acc_steps
            
            # 2. VQ Codebook Losses
            # Note: SPD_VQVAE might return scalar or tensor, ensuring items are extracted correctly
            c_loss = quantize_losses['codebook_loss']
            commit_loss = quantize_losses['commitment_loss']
            
            g_loss += (train_config['codebook_weight'] * c_loss / acc_steps)
            g_loss += (train_config['commitment_beta'] * commit_loss / acc_steps)
            # for lossk in quantize_losses:
            #     g_loss += (train_config[lossk] * quantize_losses[lossk] / acc_steps)
            codebook_losses.append(train_config['codebook_weight'] * c_loss.item())
            gen_losses.append(cholesky_loss.item())

            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                lr_scheduler.step()
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        val_recon_losses = []
        val_codebook_losses = []
        val_gen_losses = []
        val_losses = []
        for im, cholesky in val_data_loader:
            save_step_count += 1
            im = im.float().to(device)[:, None]
            with torch.no_grad():
                model_output = model(im)
            output, z, quantize_losses, L_pred = model_output
            # --- Image/Matrix Saving Logic ---
            if save_step_count % image_save_steps == 0 or save_step_count == 1:
                sample_size = min(8, im.shape[0])
                
                # Reshape/Clamp for visualization
                # Correlation matrices are [-1, 1], so this normalization is correct for PNG saving
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                
                save_dir = os.path.join(train_config['task_name'], f'{savetag}vqvae_autoencoder_samples')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                img.save(os.path.join(save_dir, 'val_current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # 1. Reconstruction Loss (MSE)
            recon_loss = recon_criterion(output, im) 
            val_recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            cholesky_loss = recon_criterion(L_pred, cholesky.to(device))
            
            # g_loss = recon_loss + cholesky_loss / acc_steps
            g_loss = cholesky_loss / acc_steps
            
            # 2. VQ Codebook Losses
            # Note: SPD_VQVAE might return scalar or tensor, ensuring items are extracted correctly
            c_loss = quantize_losses['codebook_loss']
            commit_loss = quantize_losses['commitment_loss']
            
            g_loss += (train_config['codebook_weight'] * c_loss / acc_steps)
            g_loss += (train_config['commitment_beta'] * commit_loss / acc_steps)
            # for lossk in quantize_losses:
            #     g_loss += (train_config[lossk] * quantize_losses[lossk] / acc_steps)
            val_codebook_losses.append(train_config['codebook_weight'] * c_loss.item())
            val_gen_losses.append(cholesky_loss.item())

            val_losses.append(g_loss.item())
            # g_loss.backward()
        
        log_string = (
            'Finished epoch: {} | Train Recon Loss : {:.4f} | Train Codebook : {:.4f} | '
            'Train Cholesky Loss : {:.4f} | Val Recon Loss : {:.4f} | Val Codebook : {:.4f} | '
            'Val Cholesky Loss : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(recon_losses),
                np.mean(codebook_losses),
                np.mean(gen_losses) if len(gen_losses) > 0 else 0,
                np.mean(val_recon_losses),
                np.mean(val_codebook_losses),
                np.mean(val_gen_losses) if len(val_gen_losses) > 0 else 0,
                
            )
        )
        print(log_string)
        if np.mean(val_losses) < best_loss:
            best_model = {k: v.cpu() for k,v in model.state_dict().items()}
    torch.save(best_model, os.path.join(train_config['task_name'],
                                                f'{savetag}'+train_config['vqvae_autoencoder_ckpt_name']))
    
    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/icd2fc_image_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)