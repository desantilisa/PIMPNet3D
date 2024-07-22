#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:18:57 2023

@author: lisadesanti
"""

import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import monai.transforms as transforms
import torchvision
from plot_utils import plot_3d_slices, plot_rgb_slices, generate_rgb_array, plot_atlas_overlay
import random



def get_patch_size(args):

    patch_z = round(args.img_shape[0]/args.dshape)
    patch_y = round(args.img_shape[1]/args.hshape)
    patch_x = round(args.img_shape[2]/args.hshape)
    
    patchsize = (patch_z, patch_y, patch_x)
    skip_z = round((args.img_shape[0] - patch_z) / (args.dshape-1))
    skip_y = round((args.img_shape[1] - patch_y) / (args.hshape-1))
    skip_x = round((args.img_shape[2] - patch_x) / (args.wshape-1))
    
    return patchsize, skip_z, skip_y, skip_x


# convert latent location to coordinates of image patch
def get_img_coordinates(slices, rows, cols, softmaxes_shape, patchsize, skip_z, skip_y, skip_x, d_idx, h_idx, w_idx):

    d_min = d_idx*skip_z
    d_max = min(slices, d_idx*skip_z + patchsize[0])
    h_min = h_idx*skip_y
    h_max = min(rows, h_idx*skip_y + patchsize[1])
    w_min = w_idx*skip_x
    w_max = min(cols, w_idx*skip_x + patchsize[2])                                    
    
    if d_idx == softmaxes_shape[2]-1:
        d_max = slices
    if h_idx == softmaxes_shape[3]-1:
        h_max = rows
    if w_idx == softmaxes_shape[4]-1:
        w_max = cols
        
    if d_max == slices:
        d_min = slices-patchsize[0]
    if h_max == rows:
        h_min = rows-patchsize[1]
    if w_max == cols:
        w_min = cols-patchsize[2]

    return d_min, d_max, h_min, h_max, w_min, w_max


@torch.no_grad()                    
def visualize_topk(
        net,            # trained PIMPNet
        projectloader,  # dataloader
        num_classes,    # n° classes
        device, 
        foldername, 
        args: argparse.Namespace,
        save: bool,     # save the image prototypes
        k = 10,         # n° most activated top-k images in training set
        plot = False    # plot image prototypes as montage of: (i) marked volume of interest (VOI) in the training image (ii) detailed view of the VOI
        ):
    """
    Visualises the top-k most similar image patches in the training set 
    detected with similarity > 0.1 for every relevant prototypes (class weight
    > 1e-3 for at least one classes)
    Prototypes that are not relevant to any class (all weights are zero) are 
    excluded 
    
    Return: 
        
        - topks: dictionary where:
            - key: int which identifies each relevant prototype (c_w > 1e-3)
            - value: list of tuples of the top k activated images in the 
                     training set for the corresponding prototype detected 
                     with similarity > 0.1.
                     Each tuple contains:
                         - idx of the image in the training set (unshuffled)
                         - prototype presence score
                          
       - img_prototype: dictionary where:
           - key: int which identifies each prototype of the model
           - value: list containing the topk most similar images for each
                    relevant prototype
                    
        - proto_coord: dictionary where:
            - key: int which identifies each prototype of the model
            - value: list of tuples of the coordinates in input space the 
                     top-k most similar image patches in the training set
        
    """
    
    print("Visualizing prototypes for topk...", flush = True)
    dir = os.path.join(args.log_dir, foldername)
    if save:
        if not os.path.exists(dir):
            os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    img_prototype = dict()
    proto_coord = dict()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []
        img_prototype[p] = []
        proto_coord[p] = []
    
    patchsize, skip_z, skip_y, skip_x = get_patch_size(args)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = enumerate(iter(projectloader))
    imgs = [(img, label) for img, label in zip(projectloader.dataset.img_dir, projectloader.dataset.img_labels)]
    
    # Iterate through the data
    images_seen = 0
    topks = dict()
    
    # Iterate through the training set
    for i, (xs, xs_age, ys) in img_iter:
        
        # print("Search topk activated images for each relevant prototypes,",
        #       "current image", i, flush=True)
        
        images_seen += 1
        xs, xs_age, ys = xs.to(device), xs_age.to(device), ys.to(device)

        with torch.no_grad():
            
            # Use the model to classify this batch of input data
            pfs, pages, pooled, _ = net(xs, xs_age, inference = True)
            pooled = pooled.squeeze(0)      # [ps]
            pfs = pfs.squeeze(0)            # [pimgs,d,h,w] 
            num_pimgs = pfs.shape[0]
            
            for p in range(pooled.shape[0]):
                c_weight = torch.max(classification_weights[:, p]) 
                
                # ignore prototypes that are not relevant to any class
                if c_weight > 1e-3: 
                    
                    if p not in topks.keys():
                        # initialize
                        topks[p] = []
                        
                    if len(topks[p]) < k:
                        # Add to topks:
                        # - image index in projectloader of xs
                        # - prototype presence score of p in xs
                        topks[p].append((i, pooled[p].item())) 
                        
                    else:
                        # check what are the most activated images for the 
                        # prototype p
                        topks[p] = sorted(topks[p], key = lambda tup: tup[1], reverse = True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                            
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one (since dataset 
                            # is not shuffled so later images with same scores 
                            # can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())

    alli = [] # index of input images which have the topk activation with similarity>0.1 for each relevant prototype
    prototypes_not_used = []
    
    # Check what are the prototypes with similarity > 0.1 (the ones detected in the training set) 
    for p in topks.keys():
        found = False
        
        for idx, score in topks[p]:
            alli.append(idx)
            
            if score > 0.1:  
                # in case prototypes have fewer than k well-related patches
                found = True
                
        if not found:
            prototypes_not_used.append(p)
    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    
    
    
    """ Problem! You have also indexes of age-prototypes"""
    abstained = 0
    img_iter = enumerate(iter(projectloader))

    for i, (xs, xs_age, ys) in img_iter:
        
        print("Localize each relevant prototype with similarity > 0.1 as a", "patch of the topk activated images in the training set,", i, flush=True)
        
        # shuffle is false so should lead to same order as in imgs
        if i in alli:
            
            xs, ys = xs.to(device), ys.to(device)
            
            # visualize only relevant prototypes (weights connection > 0 at least for one class)
            for p in topks.keys():
                
                # visualize only prototypes detected with similarity > 0.1
                if p not in prototypes_not_used and p < num_pimgs:
                    
                    for idx, score in topks[p]:
                        
                        if idx == i:
                            # Use the model to classify this batch of input data
                            with torch.no_grad():

                                softmaxes, pages, pooled, out = net(xs, xs_age, inference = True) # softmaxes: (1,ps,d,h,w)                 
                                outmax = torch.amax(out, dim=1)[0]  # outmax: ([1]) as projectloader's bs=1 
                                if outmax.item() == 0.:
                                    abstained += 1
                            
                            # Take the maximum per prototype in feature's space for image xs
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0) # (ps,d,h,w)
                            max_per_prototype_hw, max_idx_per_prototype_hw = torch.max(max_per_prototype, dim=1) # (ps,h,w)
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype_hw, dim=1) # (ps,w)
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) # (ps)
                            
                            c_weight = torch.max(classification_weights[:, p]) 
                            
                            # ignore prototypes that are not relevant to any class
                            if (c_weight > 1e-10) or ('pretrain' in foldername):
                                
                                # get the coordinate of the maximum in the feature's space
                                d_idx = max_idx_per_prototype_hw[p,max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]], max_idx_per_prototype_w[p]].item()
                                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]].item()
                                w_idx = max_idx_per_prototype_w[p].item()
                                img_to_open = imgs[i]
                                
                                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): 
                                    # dataset contains tuples of (img, label)
                                    img_to_open = img_to_open[0]
                                
                                img_np = np.expand_dims(np.load(img_to_open), axis=0)
                                img_min = img_np.min()
                                img_max = img_np.max()
                                img_np = (img_np-img_min)/(img_max-img_min)
                                
                                img_tensor = transforms.RepeatChannel(repeats = args.channels)(img_np)
                                img_tensor = transforms.Resize(spatial_size = (args.slices, args.rows, args.cols))(img_tensor)
                                
                                # img_tensor: shape (1, 3, slices, rows, cols)
                                img_tensor = img_tensor.unsqueeze_(0) 
                                
                                ps_coord = get_img_coordinates(args.slices, args.rows, args.cols, softmaxes.shape, patchsize, skip_z, skip_y, skip_x, d_idx, h_idx, w_idx)
                                
                                d_min = ps_coord[0]
                                d_max = ps_coord[1]
                                h_min = ps_coord[2]
                                h_max = ps_coord[3]
                                w_min = ps_coord[4]
                                w_max = ps_coord[5]
                                
                                img_tensor_patch = img_tensor[0, :, d_min:d_max, h_min:h_max, w_min:w_max]
                                        
                                saved[p]+=1
                                tensors_per_prototype[p].append(img_tensor_patch.array)
                                img_prototype[p].append(img_to_open)
                                proto_coord[p].append(ps_coord)
                                

    print("Abstained: ", abstained, flush = True)
    all_tensors = []
    
    for p in range(net.module._num_prototypes):
        
        # print("Plot prototypes", p, flush=True)
        
        if saved[p] > 0:
            
            for img_name, tensor, ps_coord in zip(img_prototype[p], tensors_per_prototype[p], proto_coord[p]):
                 
                img_np = np.expand_dims(np.load(img_name), axis=0)
                img_min = img_np.min()
                img_max = img_np.max()
                img_np = (img_np-img_min)/(img_max-img_min)
                
                img_tensor = transforms.RepeatChannel(repeats = args.channels)(img_np)
                img_tensor = transforms.Resize(spatial_size = (args.slices, args.rows, args.cols))(img_tensor)
                img_tensor = img_tensor.unsqueeze_(0) # shape: (1, 3, slices, rows, cols)
                
                d_min = ps_coord[0]
                d_max = ps_coord[1]
                h_min = ps_coord[2]
                h_max = ps_coord[3]
                w_min = ps_coord[4]
                w_max = ps_coord[5]
                
                # Create a binary mask for the cube's edges
                edges_mask = torch.zeros_like(img_tensor)
                erosion_mask = torch.zeros_like(img_tensor)
                
                edges_mask[:, 0, d_min:d_max+1, h_min:h_max+1, w_min:w_max+1] = 1
                erosion_mask[:, 0, d_min+1:d_max, h_min+2:h_max-2, w_min+2:w_max-1] = 1
                edges_mask = edges_mask - erosion_mask
                edges_mask = (edges_mask > 0).to(dtype=torch.bool)
                
                img_tensor[edges_mask] = 1.
                image = img_tensor.detach().cpu().numpy() # shape: (1, 3, slices, rows, cols)
                
                if plot:
                    plot_rgb_slices(image[0,:,:,:,:], title = "Prototype%s"%str(p), num_columns = 10, bottom=True)   
                    plot_3d_slices(tensor[0,:,:,:], title = "Prototype %s"%str(p), num_columns = 6, bottom=True)
                    plot_atlas_overlay(tensor[0,:,:,:], ps_coord, num_columns = 6,)
                    
                if save:
                    print("Poi vi salvo")
                    # np.save(os.path.join(dir, ps_name), image[0,:,:,:,:])
                    # np.save(os.path.join(dir, ps_patch_name), tensor[:,:,:,:])
                        
                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]
    return topks, img_prototype, proto_coord



def plot_local_explanation(xs, local_explanation, title=""):
    """
    Mark all the detected relevant prototypes in xs with a volume of 
    interest 
    
    Args:
        - xs: torch.Tensor, shape (bs,ch,D,H,W)
        - local_explanation: dict containing all the detected prototypes for 
          the input passed, where:
              -
              - key: int, index which identity the detected prototype
              - value: tuple containing:
                  - (dmin,dmax,hmin,hmax,wmin,wmax): tuple of the coordinates
                    in input image of the detected prototype
                  - simweight: contribution of the detected prototype to the
                    class predicted
         - title: str """
    
    num_ps = len(local_explanation.keys())
    rgb_colors = generate_rgb_array(num_ps)
    ps_scores = []

    for i, (ps_idx, ps) in enumerate(local_explanation.items()):
        
        ps_coord = ps[0]
        ps_score = ps[1]
        ps_scores.append((ps_score, rgb_colors[i]))
        
        d_min = ps_coord[0]
        d_max = ps_coord[1]
        h_min = ps_coord[2]
        h_max = ps_coord[3]
        w_min = ps_coord[4]
        w_max = ps_coord[5]
        
        # Create a binary mask for the cube's edges
        edges_mask_r = torch.zeros_like(xs)
        erosion_mask_r = torch.zeros_like(xs)
        
        edges_mask_g = torch.zeros_like(xs)
        erosion_mask_g = torch.zeros_like(xs)
        
        edges_mask_b = torch.zeros_like(xs)
        erosion_mask_b = torch.zeros_like(xs)
        
        edges_mask_r[:, 0, d_min:d_max+1, h_min:h_max+1, w_min:w_max+1] = 1
        erosion_mask_r[:, 0, d_min+1:d_max, h_min+2:h_max-2, w_min+2:w_max-1] = 1
        edges_mask_r = edges_mask_r - erosion_mask_r
        edges_mask_r = (edges_mask_r > 0).to(dtype=torch.bool)
        
        edges_mask_g[:, 1, d_min:d_max+1, h_min:h_max+1, w_min:w_max+1] = 1
        erosion_mask_g[:, 1, d_min+1:d_max, h_min+2:h_max-1, w_min+2:w_max-1] = 1
        edges_mask_g = edges_mask_g - erosion_mask_g
        edges_mask_g = (edges_mask_g > 0).to(dtype=torch.bool)
        
        edges_mask_b[:, 2, d_min:d_max+1, h_min:h_max+1, w_min:w_max+1] = 1
        erosion_mask_b[:, 2, d_min+1:d_max, h_min+2:h_max-1, w_min+2:w_max-1] = 1
        edges_mask_b = edges_mask_b - erosion_mask_b
        edges_mask_b = (edges_mask_b > 0).to(dtype=torch.bool)
        
        xs[edges_mask_r] = rgb_colors[i][0]
        xs[edges_mask_g] = rgb_colors[i][1]
        xs[edges_mask_b] = rgb_colors[i][2]
        
        
    plot_rgb_slices(np.array(xs[0,:,:,:,:]), title=title, legend=ps_scores)


