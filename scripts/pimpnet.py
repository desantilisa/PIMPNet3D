#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:25:34 2024

@author: lisadesanti
"""


import argparse
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from videoresnet_features import video_resnet18_features
from resnet_features import resnet_18_3d_features
from convnext_features import convnext_tiny_3d_features



class PIMPNet(nn.Module):
    
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 proto_age_net: nn.Module,
                 classification_layer: nn.Module
                 ):
        
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers        # Softmax over ps
        self._proto_age_net = proto_age_net # Age prototypes layer
        self._pool = pool_layer             # AdaptiveMaxPooling3D -> Flattening
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, xs_age, inference=False, print_age=False, visualize=False):
        
        img_features = self._net(xs) 
        proto_img_features = self._add_on(img_features) # (bs,ps,d,h,w)
        img_pooled = self._pool(proto_img_features) # (bs,ps,1,1,1) -> (bs,ps)
        proto_age, age_sim = self._proto_age_net(xs_age) # (bs,ps_age)

        if inference:
            if print_age:
                print("Age prototypes ", proto_age)
                print("Age similarity: ", age_sim)
            # During inference:
            #   - Use only the most activated age prototype
            #   - Ignore all prototypes that have 0.1 similarity or lower
            max_values, _ = torch.max(age_sim, dim=1, keepdim=True)
            mask = torch.eq(age_sim, max_values)
            clamped_age_sim = torch.where(mask, age_sim, torch.zeros_like(age_sim))
            clamped_img_pooled = torch.where(img_pooled < 0.1, 0., img_pooled) # (bs,ps)
            # Concatenates Image and Age prototype presence scores
            clamped_pooled = torch.cat((clamped_img_pooled, clamped_age_sim), dim=1) # image + age prototypes presence scores
            clamped_pooled = torch.where(clamped_pooled < 0.1, 0., clamped_pooled) # (bs,ps)
            out = self._classification(clamped_pooled) # (bs, num_classes)
            if visualize: 
                return proto_img_features, proto_age, clamped_img_pooled, out
            else: 
                return proto_img_features, proto_age, clamped_pooled, out
        
        else:
            pooled = torch.cat((img_pooled, age_sim), dim=1) # image + age prototypes presence scores
            out = self._classification(pooled) # (bs*2, num_classes) 
            return proto_img_features, proto_age, pooled, out
        
        
base_architecture_to_features = {
    'resnet3D_18_kin400': video_resnet18_features,
    'resnet3D_18': resnet_18_3d_features,
    'convnext3D_tiny_imgnet': convnext_tiny_3d_features,
    'convnext3D_tiny': convnext_tiny_3d_features,
    }


# adapted from 
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    
    """
    Applies a linear transformation to the incoming data with non-negative 
    weights` """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device = None, 
                 dtype = None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad = True))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)
    
    
    
class ProtoAgeSimilarity(nn.Module):
    
    """
    Computes a similarity measurement between input age and the leaned age 
    prototypes """
    
    def __init__(self, num_proto_age):
        super(ProtoAgeSimilarity, self).__init__()
        self.num_proto_age = num_proto_age
        
        # Initialize trainable tensors
        self.proto_age = nn.Parameter(
            torch.linspace(40, 90, steps=num_proto_age), requires_grad = True)

    def forward(self, xs_age):
        t = 4
        s = 8
        # Compute similarity distance between input and each age prototype
        age = xs_age.unsqueeze(1)
        age = age.expand(xs_age.shape[0], self.num_proto_age).clone()
        bs_proto_age = self.proto_age.unsqueeze(0).expand(xs_age.size(0), -1)
        #bs_proto_age = torch.round(bs_proto_age) # discretize ages 
        age_sim = 1/torch.sqrt(1 + ((age - bs_proto_age)/t)**(2*s))
        return self.proto_age, age_sim 
    
    

def get_network(num_classes: int, args: argparse.Namespace): 
    
    if args.net == 'resnet3D_18':
        img_features = base_architecture_to_features[args.net](sample_input_D = args.slices, sample_input_H = args.rows, sample_input_W = args.cols, num_classes = args.out_shape)
    
    elif args.net == 'convnext3D_tiny_imgnet':
        img_features = base_architecture_to_features[args.net](pretrained = not args.disable_pretrained, in_chan = args.channels, pretrained_mode = 'imagenet')
        
    else:
        img_features = base_architecture_to_features[args.net](pretrained = not args.disable_pretrained)
    
    features_name = str(img_features).upper()
    
    if 'next' in args.net:
        features_name = str(args.net).upper()
        
    if features_name.startswith('VIDEO') or features_name.startswith('RES') or features_name.startswith('CONV'):
        first_add_on_layer_in_channels = [i for i in img_features.modules() if isinstance(i, nn.Conv3d)][-1].out_channels

    else:
        raise Exception('other base architecture NOT implemented')
    
    if args.num_features == 0:
        num_img_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_img_prototypes, flush=True)
        add_on_layers = nn.Sequential(nn.Softmax(dim=1),)# softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1 
        
    else:
        num_img_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_img_prototypes, ". Extra 1x1x1 conv layer added. Not recommended.", flush=True)
        
        add_on_layers = nn.Sequential(
            nn.Conv3d( in_channels = first_add_on_layer_in_channels, out_channels = num_img_prototypes, kernel_size = 1, stride = 1, padding = 0, bias = True), 
            nn.Softmax(dim=1),  
            ) # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool3d(output_size=(1,1,1)), # dim: (bs,ps,1,1,1) 
        nn.Flatten()                               # dim: (bs,ps)
        ) 
    
    num_age_prototypes = args.num_age_prototypes
    proto_age_layer = ProtoAgeSimilarity(num_age_prototypes)
    num_prototypes = num_img_prototypes + num_age_prototypes
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return img_features, add_on_layers, pool_layer, proto_age_layer, classification_layer, num_prototypes






