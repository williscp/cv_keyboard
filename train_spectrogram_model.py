import torch
import os
import numpy as np
import cv2
import time
import math

from config import Config
from utils import print_accuracy
from dataset import SpectrogramDataset
from models.spectrogram_model import SpectrogramModel, SpectrogramConvModel

configs = Config()

train_set = SpectrogramDataset(configs, data_dir='data/spectrograms', label_path='labels/train.csv')
val_set = SpectrogramDataset(configs, data_dir='data/spectrograms/val', label_path='labels/val.csv')

#train_set.get_mean() # to calculate data mean
#train_set.get_class_weights() # to calculate class weights 

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

"""
Train FC model

"""

model = SpectrogramModel(configs).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
class_weights = torch.tensor(configs.class_weights, dtype=torch.float).to("cuda")
losses = []

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch 
        T = data.shape[-1]
                    
        preds = model(data)
                
        loss = torch.nn.functional.cross_entropy(preds, label, weight=class_weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        print(losses[-1])
        
np.save('output/loss.npy', losses)

print("TRAINING SET")
print_accuracy(train_loader, model)
print("VALIDATION SET")
print_accuracy(val_loader, model)
        
"""

model = SpectrogramConvModel(configs).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
class_weights = torch.tensor(configs.class_weights, dtype=torch.float).to("cuda")
losses = []

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch 
        T = data.shape[-1]

        data = data 
        label = label
        
        #print(data.shape)

        preds = model(data)

        #print(preds.shape)

        loss = torch.nn.functional.cross_entropy(preds, label, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(losses[-1])
        
np.save('output/loss.npy', losses)

print("TRAINING SET")
print_accuracy(train_loader, model)
print("VALIDATION SET")
print_accuracy(val_loader, model)
"""
    