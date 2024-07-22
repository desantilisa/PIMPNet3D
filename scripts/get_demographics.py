#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:08:12 2024

@author: lisadesanti
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas
import torch
from utils import get_args
from make_mm_dataset import get_dataloaders



#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"resnet3D_18", 3:"convnext3D_tiny_imgnet", 4:"convnext3D_tiny"}

current_fold = 1
net = backbone_dic[1]
task_performed = "Test MMPIPNet"

args = get_args(current_fold, net, task_performed)
metadata_path = args.metadata_path

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Get Dataloaders

dataloaders = get_dataloaders(args)

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]

dataset_info = trainloader_normal.dataset.dataset_info

metadata = pandas.read_csv(metadata_path)

all_ages = []
all_ages_ad = []
all_ages_mci = []
all_ages_cn = []
train_ages = []
train_ages_ad = []
train_ages_mci = []
train_ages_cn = []

for subj in dataset_info['train_subj']:
    
    age = metadata.loc[metadata['Subject'] == subj]['Age']
    age = age.to_numpy()[0]
    all_ages.append(age)
    train_ages.append(age)
    
    if metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'CN':
        all_ages_cn.append(age)
        train_ages_cn.append(age)
        
    elif metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'AD':
        all_ages_ad.append(age)
        train_ages_ad.append(age)
        
    else:
        all_ages_mci.append(age)
        train_ages_mci.append(age)
    
    
    
for subj in dataset_info['val_subj']:
    
    age = metadata.loc[metadata['Subject'] == subj]['Age']
    age = age.to_numpy()[0]
    all_ages.append(age)
    
    if metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'CN':
        all_ages_cn.append(age)
        
    elif metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'AD':
        all_ages_ad.append(age)
        
    else:
        all_ages_mci.append(age)
    


for subj in dataset_info['test_subj']:
    
    age = metadata.loc[metadata['Subject'] == subj]['Age']
    age = age.to_numpy()[0]
    all_ages.append(age)
    
    if metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'CN':
        all_ages_cn.append(age)
        
    elif metadata.loc[metadata['Subject'] == subj]['Group'].to_numpy()[0] == 'AD':
        all_ages_ad.append(age)
        
    else:
        all_ages_mci.append(age)
        
  
print("--- All dataset")
print("Mean: ", np.array(all_ages).mean(), " STD: ", np.array(all_ages).std())
print("N°: ", len(all_ages))
print("Min: ", np.array(all_ages).min(), "Max: ", np.array(all_ages).max())
print("CN")
print("Mean: ", np.array(all_ages_cn).mean(), "STD: ", np.array(all_ages_cn).std())
print("N°: ", len(all_ages_cn))
print("Min: ", np.array(all_ages_cn).min(), "Max: ", np.array(all_ages_cn).max())
print("AD")
print("Mean: ", np.array(all_ages_ad).mean(), "STD: ", np.array(all_ages_ad).std())
print("N°: ", len(all_ages_ad))
print("Min: ", np.array(all_ages_ad).min(), "Max: ", np.array(all_ages_ad).max())
print("--- Trainset ")
print("Mean: ", np.array(train_ages).mean(), "STD: ", np.array(train_ages).std())
print("N°: ", len(train_ages))
print("Min: ", np.array(train_ages).min(), "Max: ", np.array(train_ages).max())
print("CN")
print("Mean: ", np.array(train_ages_cn).mean(), "STD: ", np.array(train_ages_cn).std())
print("N°: ", len(train_ages_cn))
print("Min: ", np.array(train_ages_cn).min(), "Max: ", np.array(train_ages_cn).max())
print("AD")
print("Mean: ", np.array(train_ages_ad).mean(), "STD: ", np.array(train_ages_ad).std())
print("N°: ", len(train_ages_ad))
print("Min: ", np.array(train_ages_ad).min(), "Max: ", np.array(train_ages_ad).max())


