import yaml
import argparse
import torch
from torch import nn
import random
import torchvision
import os
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.icdfc_dataset import ICDFCDataset, FCICDDataset
from sklearn.metrics import balanced_accuracy_score
from event_complete_tokenizer import tokenizer
from lcm_models import 
class Discriminator(nn.Module):
    def __init__(self, feat_dim=1024, node_sz=116, nlayer=1, aggr='learn', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlayer = nlayer
        self.nicd = 2
        self.nage = 60
        self.net = [nn.Linear(node_sz, feat_dim)]
        # for _ in range(nlayer):
        #     self.net.append(nn.Linear(feat_dim, feat_dim))
        #     self.net.append(nn.LeakyReLU())
        
        self.net = nn.ModuleList(self.net)
        self.backbone = BNDecoder(hiddim, nclass=24, node_sz=116, nlayer=32, head_num=8, finetune=True, finetune_nclass=self.nicd+self.nage, finetune_tokenid=torch.arange(self.nicd+self.nage)).to(device)
        self.backbone.load_state_dict(torch.load(f'/cns/USERS/ziquanw/brain_network_decoder/model_weights/none_decoder32_adni-hcpa-hcpya-abide-ppmi-taowu-neurocon_boldwin500_FCFC/head_fold0_hcpaBest-y_2025-10-09-11-33-08-488037.pt', map_location='cpu'), strict=False)
        self.head1 = nn.Linear(feat_dim, self.nicd)
        self.head2 = nn.Linear(feat_dim, self.nage)

        self.aggr = aggr
        if aggr == 'learn':
            self.pool = nn.Sequential(
                nn.Linear(node_sz, feat_dim), 
                nn.LeakyReLU(),
                nn.Linear(feat_dim, feat_dim), 
                nn.LeakyReLU(),
                nn.Linear(feat_dim, 1), 
            )
    
    def forward(self, x):
        B, N = x.shape[:2]
        for net in self.net:
            x = net(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        if self.aggr == 'learn':
            y1 = self.pool(y1.view(B, N, y1.shape[-1]).transpose(-1, -2))[..., 0]
            y2 = self.pool(y2.view(B, N, y2.shape[-1]).transpose(-1, -2))[..., 0]

        return y1, y2


def train(args):
    device = 'cuda:0'
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed_all(seed)
        
    # Create the dataset
    # im_dataset_cls = {
    #     'mnist': MnistDataset,
    #     'celebhq': CelebDataset,
    #     'ICDFC': ICDFCDataset,
    #     'FCICD': FCICDDataset,
    # }.get(dataset_config['name'])
    
    im_dataset = FCICDDataset(split='train', onlyrest=True,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    print(im_dataset.cn_num, im_dataset.icd_num)
    print(np.bincount(im_dataset.delta_ages).tolist())
    # exit()
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)

    im_dataset = FCICDDataset(split='val', onlyrest=True,
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    val_data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=False)


    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # CE Loss
    disc_criterion = torch.nn.CrossEntropyLoss()
    discriminator = Discriminator().to(device)
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    acc_steps = train_config['autoencoder_acc_steps']

    step_count = 0
    best_acc = 0
    patience = 0
    max_patience = int(num_epochs*0.1)
    for epoch_idx in range(num_epochs):
        if patience > max_patience: break
        disc_losses1 = []
        disc_losses2 = []
        accs1 = []
        accs2 = []
        optimizer_d.zero_grad()
        
        # Wrapping loader in tqdm for progress bar
        for im, label, age in data_loader:
            step_count += 1
            label = label.to(device).squeeze()
            age = age.to(device).squeeze()
            im = im.float().to(device)
            
            y1, y2 = discriminator(im)
            disc_loss1 = disc_criterion(y1, label)
            disc_loss2 = disc_criterion(y2, age)
            disc_losses1.append(disc_loss1.item())
            disc_losses2.append(disc_loss2.item())
            disc_loss = (disc_loss1+disc_loss2) / acc_steps
            disc_loss.backward()
            
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()
            
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        with torch.no_grad():
            # Wrapping loader in tqdm for progress bar
            labels = []
            ages = []
            pred1s = []
            pred2s = []
            for im, label, age in val_data_loader:
                label = label.squeeze()
                age = age.squeeze()
                im = im.float().to(device)
                
                y1, y2 = discriminator(im)
                pred1 = y1.argmax(-1).detach().cpu()
                pred2 = y2.argmax(-1).detach().cpu()
                labels.extend(label)
                ages.extend(age)
                pred1s.extend(pred1)
                pred2s.extend(pred2)
            acc1 = balanced_accuracy_score(labels, pred1s)
            acc2 = balanced_accuracy_score(ages, pred2s)
            accs1.append(acc1)
            accs2.append(acc2)
                
        
        log_string = (
            'Finished epoch: {} | Train_label_loss : {:.4f} | Train_age_loss : {:.4f} | Val_label_Acc : {:.4f} | '
            'Val_deltaage_Acc : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(disc_losses1),
                np.mean(disc_losses2),
                np.mean(accs1),
                np.mean(accs2),
            )
        )
        print(log_string)
        patience += 1
        if np.mean(accs1)+np.mean(accs2) >= best_acc:
            best_acc = np.mean(accs1)+np.mean(accs2)
            patience = 0
            torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                                'fc2icd_ckpt.pth'))
    print('Done Training...')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/icd2fc_image_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)