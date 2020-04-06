import torch
import os
import numpy as np
import cv2
import time
import math

from config import Config
from dataset import SpectrogramDataset
from models.spectrogram_model import SpectrogramModel

configs = Config()

train_set = SpectrogramDataset(configs)

#train_set.get_mean()

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

model = SpectrogramModel(configs).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
losses = []

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch 
        T = data.shape[-1]
        
        data = data.squeeze(0) # remove batch dimension 
        label = label.squeeze(0)
        preds = []
        
        for tdx in range(T):
            
            pred = model(data[:,:,:,:,tdx])
            preds.append(pred)
            
        preds = torch.stack(preds).float()
                
        loss = torch.nn.functional.cross_entropy(preds, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        print(losses[-1])
        
np.save('output/loss.npy', losses)
        
    