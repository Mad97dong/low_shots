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
    
loss_fn = nn.MSELoss()
# pre-training phase
def pre_train(n_epochs, 
              train_loader, 
              g, 
              Z, # dict of all latent codes z
              optimizer, 
              pre_decoder, 
              save_root,
              dtype):
    
    g.train()
    for epoch in range(1, n_epochs+1):
        losses = []
#         progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)
        
        for i, (Xi, _, img_list) in enumerate(train_loader):
            Xi = Variable(Xi.type(dtype))
            
            G_zi = g(img_list)
            
            loss = loss_fn(G_zi, Xi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Z[img_list] = torch.tensor(pj_l2_ball(Z[img_list].data.cpu().numpy()), requires_grad=True).type(dtype)  
            losses.append(loss.item())
        
        if epoch % 200 == 0:
            print('pre-train epoch: {} \t average loss: {:.6f}'.format(epoch, np.mean(losses[-99:])))
        
#             progress.set_postfix({'loss': np.mean(losses[-99:])})
#             progress.update()

#         progress.close()
    
        # visualize reconstructions on training data
#         G_zi = g(train_image_index)
#         if epoch%10 == 0:
#             imsave('./low_shots_pk/%s_epoch_%03d.png' % ('pk', epoch), 
#                 make_grid(G_zi.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))
        
    # save the network parameter 
    torch.save(pre_decoder.state_dict(), save_root + ('dcgan_pre_%d_shots.pt' % Z.size(0)))
    torch.save(Z, save_root + "Z.pt")
   
    return g, Z

# training phase: train z + jointly train
def train(n_epochs, 
          img_noisy_var, # target image, noisy
          train_g, 
          train_decoder,
          target_z,  
          joint_train,
          learning_rate,
          dtype,
          mask=None,
          apply_f=None, # compressive sensing
          ):

    train_g.train()
    # prepare
    # train_z (joint_train=False)
    # joint train (joint_train=True)
    for param in train_decoder.parameters():
        param.requires_grad = joint_train
        
    optimizer = torch.optim.Adam(train_g.parameters(), lr=learning_rate)
    
    # training..
    for epoch in range(1, n_epochs+1):
        losses = []
#         progress = tqdm(total=1, desc='epoch % 3d' % epoch) # for one target image
        G_z = train_g([0])
        
        # inverse problems
        if mask != None:
            loss = loss_fn(G_z * mask, img_noisy_var) 
        elif apply_f != None:      
            loss = loss_fn(apply_f(G_z), img_noisy_var) # compressive sensing
        else:
            loss = loss_fn(G_z, img_noisy_var)
              
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # project z into the unit ball
        target_z[[0]] = torch.tensor(pj_l2_ball(target_z[[0]].data.cpu().numpy()), requires_grad=True).type(dtype) 
        losses.append(loss.item())
#         progress.set_postfix({'loss': np.mean(losses[-99:])})
#         progress.update()
        
#         progress.close()
           
    # back to default network
    for p in train_decoder.parameters():
        p.requires_grad = True
 
    return target_z, G_z


def pj_l2_ball(z): # z should be numpy array
    """ project vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)

def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)
    
def sample_multivar_normal(Z, code_dim):
    Z_hat = Z.view((-1, code_dim)).data.cpu().numpy()
    mu = np.mean(Z_hat, axis=0)
    var = np.cov(Z_hat.T)
    target_z = np.reshape(np.random.multivariate_normal(mu, var), (1, code_dim)).astype(np.float32)
    target_z = torch.tensor(target_z, requires_grad=True)
    return target_z  