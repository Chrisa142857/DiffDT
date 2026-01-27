import yaml, json
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
# from dataset.mnist_dataset import MnistDataset
# from dataset.celeb_dataset import CelebDataset
from dataset.icdfc_dataset import ICDFCDataset, FCICDDataset
from torch.utils.data import DataLoader
from models.unet_cond_base import Unet
from models.vqvae import VQVAE, SPD_VQVAE
from models.cholesky_ddpm import LatentMLPDiffusion
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
# from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel, Qwen3ForCausalLM
from event_complete_tokenizer import tokenizer, tokenizer_encode
from datetime import datetime
 from accelerate import Accelerator


def train(args):
    device = 'cuda:1'
    device_text = 'cuda:5'
    accelerator = Accelerator(mixed_precision="fp16") 
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
            
    im_dataset_cls = {
        # 'mnist': MnistDataset,
        # 'celebhq': CelebDataset,
        'ICDFC': ICDFCDataset,
        'FCICD': FCICDDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train', preload_embed=True, device=device_text, text_model=text_model,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         'SPD'+train_config['vqvae_latent_dir_name']),
                                condition_config=condition_config)
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)
    
    im_dataset = im_dataset_cls(split='val', preload_embed=True, device=device_text, text_model=text_model,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         'SPD'+train_config['vqvae_latent_dir_name']),
                                condition_config=condition_config)
    
    val_data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=False)
    
    # Instantiate the unet model
    # model = Unet(im_channels=1,
    #              model_config=diffusion_model_config).to(device)
    model = LatentMLPDiffusion(model_config=diffusion_model_config).to(device)
    model.train()
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    model, optimizer, data_loader, val_data_loader = accelerator.prepare(
        model, optimizer, data_loader, val_data_loader
    )
    # Now calculate steps correctly based on sharded dataloader
    num_training_steps = num_epochs * len(data_loader)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    # Register scheduler with accelerator (handles stepping logic automatically)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    best_loss = 1e+10
    best_mweight = {}
    for epoch_idx in range(num_epochs):
        losses = []
        for data in data_loader:
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    cond_input['text'] = cond_input['text'].to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            assert np.mean(losses) != np.nan, losses[-10:]
            # print(np.mean(losses))

        
        val_losses = []
        with torch.no_grad():
            for data in val_data_loader:
                cond_input = None
                if condition_config is not None:
                    im, cond_input = data
                else:
                    im = data
                optimizer.zero_grad()
                im = im.float().to(device)
                ########### Handling Conditional Input ###########
                if 'text' in condition_types:
                    with torch.no_grad():
                        assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                        validate_text_config(condition_config)
                        cond_input['text'] = cond_input['text'].to(device)

                # Sample random noise
                noise = torch.randn_like(im).to(device)
                
                # Sample timestep
                t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
                
                # Add noise to images according to timestep
                noisy_im = scheduler.add_noise(im, noise, t)
                noise_pred = model(noisy_im, t, cond_input=cond_input)
                loss = criterion(noise_pred, noise)
                val_losses.append(loss.item())
                assert np.mean(val_losses) != np.nan, losses[-10:]
                
        accelerator.print(f'{datetime.now()}'+'Epoch:{} | LR: {:.6f} | Loss : {:.4f} | Val loss: {:.4f}'.format(
            epoch_idx + 1, lr_scheduler.get_last_lr()[0],
            np.mean(losses),
            np.mean(val_losses)))

        
        # 1. Calculate Local Average
        local_avg_val_loss = np.mean(val_losses)
        
        # 2. SYNC: Gather metrics from all GPUs to calculate Global Average
        # Convert to tensor for gathering
        val_loss_tensor = torch.tensor(local_avg_val_loss, device=accelerator.device)
        
        # Gather all values and take the mean so all GPUs have the EXACT same number
        gathered_losses = accelerator.gather(val_loss_tensor)
        global_avg_val_loss = gathered_losses.mean().item()

        # 3. Use the GLOBAL average for the check
        if global_avg_val_loss <= best_loss:
            best_loss = global_avg_val_loss
            # patience = 0
            
            # Now it is safe to sync because ALL GPUs are guaranteed to enter this block
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            if accelerator.is_main_process:
                best_mweight = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Saving best model from CPU RAM to disk...")
        torch.save(best_mweight, os.path.join(train_config['task_name'],
                                                    # 'l1loss_'+train_config['ldm_ckpt_name']))
                                                    'best_'+train_config['ldm_ckpt_name']))
    
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/icd2fc_image_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)
