import torch 
import torch.nn as nn
import scipy.signal as signal
import numpy as np

def integral_heatmap_layer(heatmaps):

    N, K, H, W,  = heatmap.shape
    
    # apply softmax:

    heatmap = heatmap.reshape(N, K, -1)
    probmap = torch.nn.functional.softmax(heatmap, dim=2)
    h_norm = probmap.reshape(N, K, H, W)

    # generate the integrals 

    x_linspace = torch.linspace(0, 1, W).repeat(N, K, H, 1).to("cuda")
    y_linspace = torch.linspace(0, 1, H).repeat(N, K, W, 1).permute(0,1,3,2).to("cuda")

    x_weights = x_linspace * h_norm 
    y_weights = y_linspace * h_norm 

    x_positions = torch.sum(torch.sum(x_weights, dim=3), dim=2).unsqueeze(-1)
    y_positions = torch.sum(torch.sum(y_weights, dim=3), dim=2).unsqueeze(-1)

    pose = torch.cat((x_positions, y_positions), dim=2)
    
    return pose

class StrokeModel(nn.Module):
    
    def __init__(self, configs):
        super(StrokeModel, self).__init__()
        
        self.input_size = configs.input_size 
        self.fc1 = nn.Linear(22 * 2, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 27)
        
    def forward(self, data):
        
        left_maps, left_bbox, right_maps, right_bbox = data 
        left_pose = integral_heatmap_layer(left_maps)
        right_pose = integral_heatmap_layer(right_maps)
        
        
        
        # change to global coordinates
        
        
        
        
        
        
        
        
        
        
        