import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from dataset.icdfc_dataset import ICDFCDataset, FCICDDataset
from models.vqvae import VQVAE, SPD_VQVAE

device = 'cuda:2'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    

def infer(args):
    latent_tag = 'DualSPD_thr25'
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        # 'mnist': MnistDataset,
        # 'celebhq': CelebDataset,
        'ICDFC': ICDFCDataset,
        'FCICD': FCICDDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train', onlyrest=True, use_latents=False,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)
    
    val_im_dataset = im_dataset_cls(split='val', onlyrest=True, use_latents=False,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    val_data_loader = DataLoader(val_im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = 50 #train_config['num_samples']
    ngrid = 10 # train_config['num_grid_rows']
    
    idxs = torch.randint(0, len(val_im_dataset) - 1, (num_images,))
    ims = torch.cat([val_im_dataset[idx][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    im1 = decompose_fc(ims, dec_bg=False, threshold=25)#[:, None]
    im2 = decompose_fc(ims, dec_bg=True, threshold=25)#[:, None]

    # model = SPD_VQVAE(116, model_config=autoencoder_config).to(device)
    # model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                 'SPD'+train_config['vqvae_autoencoder_ckpt_name']),
    #                                  map_location=device))
    
    model1 = SPD_VQVAE(input_dim=116,
                          model_config=autoencoder_config,
                       num_latents=256,).to(device)
    model1.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    latent_tag+'_m1'+train_config['vqvae_autoencoder_ckpt_name']),
                                     map_location=device))
    # LPIPS (Perceptual Loss) is designed for Natural Images (RGB). 
    # It is meaningless/harmful for Correlation Matrices.
    model2 = SPD_VQVAE(input_dim=116,
                      model_config=autoencoder_config,
                       num_latents=16, # 
                       codebook_size=16 # 
                      ).to(device)
    model2.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    latent_tag+'_m2'+train_config['vqvae_autoencoder_ckpt_name']),
                                     map_location=device))
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        
        encoded_output1, _, _ = model1.encode(im1)
        decoded_output1 = model1.decode(encoded_output1)[0]
        encoded_output2, _, _ = model2.encode(im2)
        decoded_output2 = model2.decode(encoded_output2)[0]
        encoded_output1 = torch.clamp(encoded_output1, -1., 1.)
        encoded_output1 = (encoded_output1 + 1) / 2
        encoded_output2 = torch.clamp(encoded_output2, -1., 1.)
        encoded_output2 = (encoded_output2 + 1) / 2
        decoded_output1 = torch.clamp(decoded_output1, -1., 1.)
        decoded_output1 = (decoded_output1 + 1) / 2
        decoded_output2 = torch.clamp(decoded_output2, -1., 1.)
        decoded_output2 = (decoded_output2 + 1) / 2
        decoded_output = reverse_decomp(decoded_output1.squeeze(), decoded_output2.squeeze())
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        im1 = (im1 + 1) / 2
        im2 = (im2 + 1) / 2
        ims = (ims + 1) / 2
        encoder_grid = make_grid(encoded_output1.cpu()[:, None], nrow=ngrid)
        decoder_grid = make_grid(decoded_output1.cpu(), nrow=ngrid)
        input_grid = make_grid(im1.cpu()[:, None], nrow=ngrid)

        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        input_grid.save(os.path.join(train_config['task_name'], latent_tag+'input_samples_m1.png'))
        encoder_grid.save(os.path.join(train_config['task_name'], latent_tag+'encoded_samples_m1.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], latent_tag+'reconstructed_samples_m1.png'))
        
        encoder_grid = make_grid(encoded_output2.cpu()[:, None], nrow=ngrid)
        decoder_grid = make_grid(decoded_output2.cpu(), nrow=ngrid)
        input_grid = make_grid(im2.cpu()[:, None], nrow=ngrid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        input_grid.save(os.path.join(train_config['task_name'], latent_tag+'input_samples_m2.png'))
        encoder_grid.save(os.path.join(train_config['task_name'], latent_tag+'encoded_samples_m2.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], latent_tag+'reconstructed_samples_m2.png'))
        
        decoder_grid = make_grid(decoded_output.cpu()[:, None], nrow=ngrid)
        input_grid = make_grid(ims.cpu()[:, None], nrow=ngrid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        input_grid.save(os.path.join(train_config['task_name'], latent_tag+'input_samples.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], latent_tag+'reconstructed_samples.png'))
        
        # exit()
        if train_config['save_latents']:
            # save Latents (but in a very unoptimized way)
            latent_path = os.path.join(train_config['task_name'], latent_tag+train_config['vqvae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(train_config['task_name'], latent_tag+train_config['vqvae_latent_dir_name'],
                                                   '*.pkl'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))
            
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, (im, beh) in enumerate(tqdm(data_loader)):
                im1 = decompose_fc(im.float().to(device), dec_bg=False, threshold=25)
                im2 = decompose_fc(im.float().to(device), dec_bg=True, threshold=25)
                encoded_output1, _, _ = model1.encode(im1)
                encoded_output2, _, _ = model2.encode(im2)
                encoded_output = torch.cat([encoded_output1, encoded_output2], 1)
                # encoded_output, _, _ = model.encode(im.float().to(device))
                fname_latent_map[im_dataset.image_names[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                if (count+1) % 1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    f'train{part_count:06d}.pkl'), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                   f'train{part_count:06d}.pkl'), 'wb'))
            
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, (im, beh) in enumerate(tqdm(val_data_loader)):
                im1 = decompose_fc(im.float().to(device), dec_bg=False, threshold=25)
                im2 = decompose_fc(im.float().to(device), dec_bg=True, threshold=25)
                encoded_output1, _, _ = model1.encode(im1)
                encoded_output2, _, _ = model2.encode(im2)
                encoded_output = torch.cat([encoded_output1, encoded_output2], 1)
                # encoded_output, _, _ = model.encode(im.float().to(device))
                fname_latent_map[val_im_dataset.image_names[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                if (count+1) % 1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    f'val{part_count:06d}.pkl'), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                   f'val{part_count:06d}.pkl'), 'wb'))
            
            
            print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/icd2fc_image_cond.yaml', type=str)
    args = parser.parse_args()
    infer(args)
