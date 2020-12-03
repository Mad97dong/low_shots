from __future__ import print_function
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
import copy
from helpers import *

# == decoder part == #
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

    
def set_to(tensor,mtx):
    if not len(tensor.shape)==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if i == j:
                tensor[i,j] = np_to_tensor(mtx)
            else:
                tensor[i,j] = np_to_tensor(np.zeros(mtx.shape))
    return tensor

def conv2(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if kernel_size != 4:
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    else:
        padder = nn.ReflectionPad2d( (1,0,1,0) )
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=1, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)
    
#model 0 
def decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad, bias=bias))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad,bias=bias))        
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
#         model.add(nn.Tanh())
    
    return model

#model 1/3: upsample_mode='bilinear'/'none'
def fixed_decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False,
        mtx = np.array( [[1,3,3,1] , [3,9,9,3], [3,9,9,3], [1,3,3,1] ] )*1/16.
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            # those will be fixed
            model.add(conv2( num_channels_up[i], num_channels_up[i],  4, 1, pad=pad))  
            # those will be learned
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # those will be fixed
            model.add(conv2( num_channels_up[i], num_channels_up[i],  4, 1, pad=pad))  
            # those will be learned
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))      
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
#         model.add(nn.Sigmoid())
        model.add(nn.Tanh())
        
        
    # set filters to fixed and then set the gradients to zero
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if(m.kernel_size == mtx.shape):
                m.weight.data = set_to(m.weight.data,mtx)
                for param in m.parameters():
                    param.requires_grad = False
    
    return model

#model 2/4: upsample_mode='bilinear'/'none'
def deconv_decoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False,
        filter_size=4,
        stride=2,
        padding=[0]*5,
        output_padding=0
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    padding = padding + [padding[-1],padding[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add( nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1], filter_size, stride=stride, padding=padding[i], output_padding=output_padding, bias=bias, groups=1, dilation=1) )
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            model.add( nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1], filter_size=filter_size, stride=stride, padding=0, output_padding=output_padding, bias=bias, groups=1, dilation=1) )
    
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
        model.add(nn.Tanh())
    
    return model


# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np