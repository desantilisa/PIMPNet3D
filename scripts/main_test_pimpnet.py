#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:21:00 2024

@author: lisadesanti
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score


from utils import get_args
from make_mm_dataset import get_dataloaders
from model_builder import load_trained_pimpnet
from test_model import eval_pimpnet
from test_model import get_local_explanations
from test_model import eval_local_explanations
from vis_pipnet import visualize_topk



#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"resnet3D_18",  3:"convnext3D_tiny_imgnet", 4:"convnext3D_tiny"}

current_fold = 1
net = backbone_dic[1]
task_performed = "Test PIMPNet"

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Get Dataloaders for the current_fold

dataloaders = get_dataloaders(args)

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]
    
    
#%% Evaluate 3D-PIPNet trained for the current_fold

print("------", flush = True)
print("PIMPNet performances @fold: ", current_fold, flush = True)
    
pimpnet = load_trained_pimpnet(args)
pimpnet.eval()

# Forward one batch through the backbone to get the latent output size
with torch.no_grad():
    xs1, _, xs1_age, _ = next(iter(trainloader))
    xs1 = xs1.to(device)
    xs1_age = xs1_age.to(device)
    proto_features, _, _, _ = pimpnet(xs1, xs1_age)
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape # needed for calculating image patch size
    args.hshape = hshape # needed for calculating image patch size
    args.dshape = dshape # needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)


#%% Get the Global Explanation

topks, img_prototype, proto_coord = visualize_topk(
    pimpnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'visualised_prototypes_topk', 
    args,
    save=False,
    k=1,
    plot=True)

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        found = False
        for (i_id, score) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(pimpnet.module._classification.weight[:,prot])
            set_to_zero.append(prot)
    print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)

print("Classifier weights: ", pimpnet.module._classification.weight, flush = True)
print("Classifier weights nonzero: ", pimpnet.module._classification.weight[pimpnet.module._classification.weight.nonzero(as_tuple=True)], (pimpnet.module._classification.weight[pimpnet.module._classification.weight.nonzero(as_tuple=True)]).shape, flush = True)
print("Classifier bias: ", pimpnet.module._classification.bias, flush = True)


# Print weights and relevant prototypes per class
relevant_ps_dic = {}
for c in range(pimpnet.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = pimpnet.module._classification.weight[c,:]
    
    for p in range(pimpnet.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append(p)
            #relevant_ps.append((p, proto_weights[p].item()))
    relevant_ps_dic[c] = relevant_ps
    
    print("Class", c, "(", 
          list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],
          "):", "has", len(relevant_ps), "relevant prototypes: ", relevant_ps,  flush = True)

    
class_shared_ps = list(set(relevant_ps_dic[0]) & set(relevant_ps_dic[1]))
print("Class shared prototypes: ", class_shared_ps)
print("Age prototypes: ", 
      pimpnet.module._proto_age_net.proto_age, flush=True)
print("CN Age Prototypes", pimpnet.module._classification.weight[0,-(1+args.num_age_prototypes):-1])
print("AD Age Prototypes", pimpnet.module._classification.weight[1,-(1+args.num_age_prototypes):-1])


#%% Evaluate PIMPNet: 
#    - Classification performances, 
#    - Explanations' size
info = eval_pimpnet(
    pimpnet, 
    testloader, 
    "notused" + str(args.epochs),
    device, 
    print_age=False)

for elem in info.items():
    print(elem)
    
local_explanations_test, y_preds_test, y_trues_test = get_local_explanations(pimpnet, testloader, device, args)


#%% Evaluate the prototypes extracted

columns=["detection_rate", "mean_pcc_d", "mean_pcc_h", "mean_pcc_w", "std_pcc_d", "std_pcc_h", "std_pcc_w","entropy", "LC"]

ps_test_evaluation = eval_local_explanations(pimpnet, local_explanations_test, device, args)

ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_mean_entropy = pd.Series(ps_test_evaluation[3])
avg_ps_purity = np.nanmean(np.array([h for h in ps_test_evaluation[3].values()]))
print("Prototype purity: ", avg_ps_purity)
ps_test_lc = pd.Series(ps_test_evaluation[4])
avg_ps_consistency = np.nanmean(np.array([h for h in ps_test_evaluation[4].values()]))
print("Prototype localization consistency: ", avg_ps_consistency)
eval_proto_test = pd.concat(
    [ps_test_detections,
     ps_test_mean_coords, 
     ps_test_std_coords,
     ps_test_mean_entropy,
     ps_test_lc], axis=1)
eval_proto_test.columns = columns 
avg_ps_background = np.nanmean(np.array([h for h in ps_test_evaluation[5].values()]))
print("Averages percentage of background: ", avg_ps_background)