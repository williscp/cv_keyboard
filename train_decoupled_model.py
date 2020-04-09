import torch
import os
import numpy as np
import cv2
import time
import math

from config import Config
from utils import print_accuracy
from dataset import SpectrogramDataset
from visualize import visualize_predictions
from models.decoupled_model import ActivationModel

configs = Config()

train_set = SpectrogramDataset(configs, label_path='labels/train.csv')
val_set = SpectrogramDataset(configs, label_path='labels/val.csv')

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
    shuffle=False,
    drop_last=False
)

model = ActivationModel(configs).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
class_weights = torch.tensor(configs.class_weights, dtype=torch.float).to("cuda")
losses = []
best_val_acc = 0
best_model = 0

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch 
        T = data.shape[-1]
        
        label = torch.clamp(label, 0, 1)

        preds = model(data)

        loss = torch.nn.functional.cross_entropy(preds, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(np.mean(losses[-100:]))
        
    val_acc = print_accuracy(val_loader, model, debug=True, activation_only=True)
    model = model.train()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = epoch
        torch.save(model.state_dict(),'saves/decoupled_model_{}.pth'.format(epoch))
                
np.save('output/loss.npy', losses)

model.load_state_dict(torch.load('saves/decoupled_model_{}.pth'.format(best_model)))

print("BEST EPOCH: {}".format(best_model))
print("TRAINING SET")
print_accuracy(train_loader, model, debug=True, activation_only=True)
print("VALIDATION SET")
print_accuracy(val_loader, model, debug=True, activation_only=True)


# Visualize output 

model.eval()
with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        data, label = batch 
        video_id, _, _, = val_set.labels[idx]
        t = val_set.t[idx]
        preds = model(data)
        preds = torch.argmax(preds, dim=1)
        
        in_file = os.path.join('data/videos','{}.mp4'.format(video_id))
        out_file = os.path.join('output/decoupled_model', '{}.avi'.format(video_id))
        
        preds = preds.detach().cpu().numpy().squeeze()
        label = label.detach().cpu().numpy().squeeze()
                
        visualize_predictions(in_file, out_file, preds, label, t)
    
    
    