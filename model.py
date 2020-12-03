import os
import plac
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.utils as vutils


code_dim = 128 # default for celebA
class dcgan(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(dcgan, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 16), 
            nn.ReLU(True), # 512 * 4 * 4
            nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),   
            nn.ReLU(True), # 256 * 8 * 8
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(nf * 4), 
            nn.ReLU(True), # 128 * 16 * 16
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2), 
            nn.ReLU(True), # 64 * 32 * 32
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf), 
            nn.ReLU(True), # 32 * 64 * 64
            nn.ConvTranspose2d(nf, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            # 3 * 128 * 128
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))
    

    
class latent_var(nn.Module):
    def __init__(self, Z):
        super(latent_var, self).__init__()
        self.Z = Parameter(Z)
        self.dim_Z = Z.size(0) # = number of low shots

    def forward(self, indices): # indices should be a list
        if max(indices) > self.dim_Z - 1:
            print('error')
        return self.Z[indices]
    
#     def pr_l2_ball(self, idx):
#         # projection 
#         tmp = torch.sqrt(torch.sum(self.Z[idx] ** 2, axis=1)).view(self.Z[idx].shape)
#         tmp = torch.max(tmp, 0.5*torch.ones(tmp.shape).to(device))
#         self.Z[idx] = self.Z[idx] / tmp
#         return self.Z[idx]


class CombinedNets(nn.Module):
    def __init__(self, LV, Decoder, n_filters=64):
        super(CombinedNets, self).__init__()
        self.Z = LV
        self.Decoder = Decoder

    def forward(self, index):
        code = self.Z(index)
        return self.Decoder(code)
    
    
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv')!=-1:
        module.weight.data.normal_(0.0, .02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, .01)
        module.bias.data.fill_(0.0)


# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np        

class IndexedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during loops.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)        
       
class Pokeman(Dataset):
    def __init__(self, imgs, names):
        self.data = imgs
        self.label = names
        
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx], idx