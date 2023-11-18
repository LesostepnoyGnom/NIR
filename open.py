# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:02:01 2023

@author: 1618047
"""


import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph

from torch.utils.tensorboard import SummaryWriter
os.makedirs('train_log', exist_ok=True)
os.makedirs('rollouts', exist_ok=True)

INPUT_SEQUENCE_LENGTH = 6
batch_size = 2
noise_std = 6.7e-4
training_steps = int(2e7)
log_steps = 5
eval_steps = 20
save_steps = 100
model_path = None # 'model425000.pth'
device = 'cuda'
# with open('data/metadata.json', 'rt') as f:
with open("train_log/**.pickle", "rb") as f:
    x = pickle.load(f)
num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
normalization_stats = {
    'acceleration': {
        'mean':torch.FloatTensor(metadata['acc_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + noise_std**2).to(device),
    }, 
    'velocity': {
        'mean':torch.FloatTensor(metadata['vel_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 + noise_std**2).to(device),
    }, 
}