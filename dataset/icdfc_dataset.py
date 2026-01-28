import glob
import os
import random
import torch
# import torchvision
import numpy as np
# from PIL import Image
from tqdm import tqdm, trange
from torch.utils.data.dataset import Dataset
import pandas as pd
from event_complete_tokenizer import tokenizer, tokenizer_encode
import pickle
from transformers import Qwen3ForCausalLM


def tokenize(txtlist, max_length=4, **kwargs):
    ids = []
    masks = []
    posids = []
    for txt in txtlist:
        input_ids, posid = tokenizer_encode(txt)
        input_ids = torch.LongTensor(input_ids)
        attn_mask = torch.ones_like(input_ids)
        posid = torch.LongTensor(posid)
        if max_length > len(input_ids):
            posid = torch.cat([posid, torch.zeros(max_length-len(input_ids))]).long()
            input_ids = torch.cat([input_ids, torch.zeros(max_length-len(input_ids))]).long()
            attn_mask = torch.cat([attn_mask, torch.zeros(max_length-len(attn_mask))]).long()
        ids.append(input_ids[:max_length])
        masks.append(attn_mask[:max_length])
        posids.append(posid[:max_length])
    ids = torch.stack(ids)
    masks = torch.stack(masks)
    posids = torch.stack(posids)
    return {
        "input_ids": ids,
        "attention_mask": masks,
        "seq_posids": posids
    }
    
def get_text_representation(text, text_tokenizer, text_model, device,
                            truncation=True,
                            padding='max_length',
                            max_length=64):
    token_output = text_tokenizer(text,
                                  truncation=truncation,
                                  padding=padding,
                                  return_attention_mask=True,
                                  max_length=max_length)
    indexed_tokens = token_output['input_ids']
    att_masks = token_output['attention_mask']
    seq_posids = token_output['seq_posids']
    tokens_tensor = torch.tensor(indexed_tokens).to(device)
    mask_tensor = torch.tensor(att_masks).to(device)
    seq_posids = torch.tensor(seq_posids).to(device)
    text_embed = text_model(
                    input_ids=tokens_tensor, 
                    attention_mask=mask_tensor, 
                    position_ids=seq_posids, output_hidden_states=True
                ).hidden_states[-1]
    return text_embed
    
def load_latents(latent_path, split='train'):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        if not fname.split('/')[-1].startswith(split): continue
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps
def robust_cholesky(matrix):
    """
    Decomposes using Eigendecomposition, clamps eigenvalues, 
    reconstructs, then runs Cholesky.
    """
    # 1. Eigendecomposition
    L, Q = torch.linalg.eigh(matrix)
    
    # 2. Clamp eigenvalues to be strictly positive
    # (min value must be > 0 for Cholesky)
    L_clamped = torch.clamp(L, min=1e-6)
    
    # 3. Reconstruct Matrix
    # M_recon = Q * diag(L) * Q^T
    M_recon = Q @ torch.diag_embed(L_clamped) @ Q.transpose(-2, -1)
    
    # 4. Now Cholesky will succeed
    return torch.linalg.cholesky(M_recon)
class ICDFCDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg', preload_embed=False, device='cuda:0', skip_future=True,
                 compute_textembed=False, use_latents=True, latent_path=None, condition_config=None, onlyrest=True):
        self.split = split
        # self.im_size = im_size
        # self.im_channels = im_channels
        # self.im_ext = im_ext
        # self.im_path = im_path
        self.latent_maps = None
        self.use_latents = use_latents
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        # self.idx_to_cls_map = {}
        # self.cls_to_idx_map ={}
        
        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']
            
        # assert not use_latents, 'Not implemented'
        
        data = pd.read_pickle('../brain_env_ukb/data/ukb-nimg_icd10_dated.pkl')
        vocab_grouped = []
        vocab = []
        for icd in data['ICD10']:
            if isinstance(icd, list):
                vocab_grouped.extend([i[:3] for i in icd])
                vocab.extend(icd)

        vocab_grouped = np.unique(vocab_grouped)
        vocab = np.unique(vocab)    
        data['ICD10_grouped'] = data['ICD10'].map(lambda x: [xi[:3] for xi in x] if isinstance(x, list) else np.nan)
        data['FC_beh'] = data['FC_name'].map(lambda x: np.array([xi.split('_')[2].split('-')[1] for xi in x] if isinstance(x, list) else np.nan))
        self.beh_book = np.unique(np.concatenate([x.reshape(-1) for x in data['FC_beh'] if not isinstance(x, float)])).tolist()
        print(self.beh_book)
        if split == 'train':
            split_data = np.memmap('../brain_env_ukb/data/delphi_train.bin', dtype=np.uint32, mode='r').reshape(-1, 3) 
        else:
            split_data = np.memmap('../brain_env_ukb/data/delphi_val.bin', dtype=np.uint32, mode='r').reshape(-1, 3)
        nan_pid = data.index[data['FC'].isna()].tolist()
        seqstep = 20
        self.image_names = []
        self.texts = []
        self.images = []
        self.masks = []
        self.behs = []
        self.fcages = []
        for i, row in tqdm(data.iterrows(), total=len(data), desc='prepare data list'):
            if i not in split_data[:, 0]: continue
            if i in nan_pid: continue
            if np.isnan(row['FC']).any(): continue
            if onlyrest and 'rest' not in row['FC_beh']: continue
            
            fcages = np.array(row['FC_age (days)'])
            if onlyrest:
                valid_idx = np.where(row['FC_beh']=='rest')[0]
            else:
                valid_idx = np.arange(len(row['FC_beh']))
                
            for fci in valid_idx:
                seq = []
                fcage = fcages[fci]
                if skip_future: 
                    endage = (fcage//365+1)
                else:
                    endage = 81
                for agey in range(25, endage):
                    agey1 = agey * 365
                    agey2 = (agey+1) * 365
                    # if (agey1 >= fcage).all() and len(seq) >= seqstep: break
                    agemask = np.logical_and(np.array(row['ICD10_age (days)']) >= agey1, 
                                np.array(row['ICD10_age (days)']) < agey2)
                    if not agemask.any():
                        seq.append('CN')
                    else:
                        x = np.array(row['ICD10_grouped'])[agemask]
                        x.sort()
                        seq.append('-'.join(list(x)))
                seqlist = [seq]
                seqlist = ['<|startoftext|> ' + ' '.join(seq) + ('' if skip_future else ' <|endoftext|>') for seq in seqlist]

                im = torch.from_numpy(row['FC'][fci]).float()
                # im[im.isnan()] = 0
                # im = torch.concat([im, torch.zeros(1, 4, im.shape[2])], 1)
                # im = torch.concat([im, torch.zeros(1, im.shape[1], 4)], 2)
                mask = torch.zeros_like(im).bool()
                # mask[:, :-4, :-4] = True
                assert not im.isnan().any()
                self.texts.append(seqlist)
                self.images.append(im)
                self.image_names.append(f'{i}-{fci}')
                self.masks.append(mask.float())
                self.behs.append(self.beh_book.index(row['FC_beh'][fci]))
                self.fcages.append(fcage)

        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path, split)
            valid_latent = True
            for k in self.image_names:
                if k not in latent_maps:
                    valid_latent = False
                    break
            if valid_latent:
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found', len(latent_maps), len(self.images))
                exit()
        self.preload_embed = preload_embed
        if preload_embed:
            if not os.path.exists(f'icd2fc/{split}_text_embed.pth') or compute_textembed:
                text_model = Qwen3ForCausalLM.from_pretrained('icd10_tokenizer_posid/qwen-icd10-complete/best_model').to(device)
                text_model.eval()
                self.text_embed = []
                # for txt in tqdm(self.texts, desc='Preload text embed'):
                for txti in trange(0, len(self.texts), 500, desc='Preload text embed'):
                    with torch.no_grad():
                        batch = []
                        for txt in self.texts[txti:txti+500]:
                            batch.extend(txt)    
                        text_condition = get_text_representation(batch, tokenize, text_model, device, max_length=text_model.config.max_position_embeddings).cpu()
    
                    self.text_embed.extend(text_condition)
                if not compute_textembed: torch.save(self.text_embed, f'icd2fc/{split}_text_embed.pth')
                
            else:
                self.text_embed = torch.load(f'icd2fc/{split}_text_embed.pth')
        if not self.use_latents:
            if not os.path.exists(f'icd2fc/{split}_cholesky_gt.pth'):
                self.cholesky_gt = []
                for img in tqdm(self.images):
                    self.cholesky_gt.append(robust_cholesky(img))
                torch.save(torch.stack(self.cholesky_gt), f'icd2fc/{split}_cholesky_gt.pth')
            else:
                self.cholesky_gt = torch.load(f'icd2fc/{split}_cholesky_gt.pth')
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            # cond_inputs['text'] = self.texts[index]
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0] if not self.preload_embed else self.text_embed[index]

        # if 'image' in self.condition_types:
        #     mask = self.masks[index]
        #     cond_inputs['image'] = mask
        
        #######################################
        
        if self.use_latents:
            # print("Not implemented, exit")
            # exit()
            latent = self.latent_maps[self.image_names[index]]
            if len(self.condition_types) == 0:
                return latent, self.cholesky_gt[index]
            else:
                return latent, cond_inputs
        else:
            # Convert input to -1 to 1 range.
            im_tensor = self.images[index]
            if len(self.condition_types) == 0:
                return im_tensor, self.cholesky_gt[index]
            else:
                return im_tensor, cond_inputs



class FCICDDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',onlyrest=True,
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        # self.im_size = im_size
        # self.im_channels = im_channels
        # self.im_ext = im_ext
        # self.im_path = im_path
        # self.latent_maps = None
        self.use_latents = False
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        # self.idx_to_cls_map = {}
        # self.cls_to_idx_map ={}
        
        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']
            
        assert not use_latents, 'Not implemented'
        
        data = pd.read_pickle('../brain_env_ukb/data/ukb-nimg_icd10_dated.pkl')
        vocab_grouped = []
        vocab = []
        for icd in data['ICD10']:
            if isinstance(icd, list):
                vocab_grouped.extend([i[:3] for i in icd])
                vocab.extend(icd)

        vocab_grouped = np.unique(vocab_grouped)
        vocab = np.unique(vocab)    
        data['ICD10_grouped'] = data['ICD10'].map(lambda x: [xi[:3] for xi in x] if isinstance(x, list) else np.nan)
        data['FC_beh'] = data['FC_name'].map(lambda x: np.array([xi.split('_')[2].split('-')[1] for xi in x] if isinstance(x, list) else np.nan))
        
        if split == 'train':
            split_data = np.memmap('../brain_env_ukb/data/delphi_train.bin', dtype=np.uint32, mode='r').reshape(-1, 3) 
        else:
            split_data = np.memmap('../brain_env_ukb/data/delphi_val.bin', dtype=np.uint32, mode='r').reshape(-1, 3)
        nan_pid = data.index[data['FC'].isna()].tolist()
        self.texts = []
        self.images = []
        self.labels = []
        self.masks = []
        self.behs = []
        self.delta_ages = []
        self.cn_num = 0
        self.icd_num = 0
        for i, row in tqdm(data.iterrows(), total=len(data), desc='prepare data list'):
            if i not in split_data[:, 0]: continue
            if i in nan_pid: continue
            if np.isnan(row['FC']).any(): continue
            if onlyrest and 'rest' not in row['FC_beh']: continue
            
            fcages = np.array(row['FC_age (days)'])
            if onlyrest:
                valid_idx = np.where(row['FC_beh']=='rest')[0]
            else:
                valid_idx = np.arange(len(row['FC_beh']))
                
            for fci in valid_idx:
                seq = []
                has_icd = False
                delta_age = 0
                fcage = fcages[fci]
                for agey in range(fcage//365, 89):
                    delta_age += 1
                    agey1 = agey * 365
                    agey2 = (agey+1) * 365
                    agemask = np.logical_and(np.array(row['ICD10_age (days)']) >= agey1, 
                                np.array(row['ICD10_age (days)']) < agey2)
                    if agemask.any():
                        x = np.array(row['ICD10_grouped'])[agemask]
                        x.sort()
                        # seq.append('-'.join(list(x)))
                        has_icd = True
                        label = 1
                        break
                if not has_icd: 
                    label = 0
                    
                im = torch.from_numpy(row['FC'][fci]).float()
                # im[im.isnan()] = 0
                # im = torch.concat([im, torch.zeros(1, 4, im.shape[2])], 1)
                # im = torch.concat([im, torch.zeros(1, im.shape[1], 4)], 2)
                mask = torch.zeros_like(im).bool()
                # mask[:, :-4, :-4] = True
                assert not im.isnan().any()
                if self.cn_num > self.icd_num and label == tokenizer['CN']: continue
                if has_icd:
                    self.icd_num += 1
                else:
                    self.cn_num += 1
                self.labels.append(label)
                # self.texts.append(seqlist)
                self.delta_ages.append(delta_age)
                self.images.append(im)
                # self.masks.append(mask.float())
                # self.behs.append(self.beh_book.index(row['FC_beh'][fci]))

                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        # cond_inputs = {}
        # if 'text' in self.condition_types:
        #     # cond_inputs['text'] = self.texts[index]
        #     cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]

        # if 'image' in self.condition_types:
        #     mask = self.masks[index]
        #     cond_inputs['image'] = mask
        #######################################
        
        # if self.use_latents:
        #     print("Not implemented, exit")
        #     exit()
        #     # latent = self.latent_maps[self.images[index]]
        #     # if len(self.condition_types) == 0:
        #     #     return latent
        #     # else:
        #     #     return latent, cond_inputs
        # else:
        #     # Convert input to -1 to 1 range.
        #     im_tensor = self.images[index]
        #     if len(self.condition_types) == 0:
        #         return im_tensor
        #     else:
        return self.images[index], self.labels[index], self.delta_ages[index]
