import torch 
import torch.nn as nn
import numpy as np

class SpectrogramModel(nn.Module):
    
    def __init__(self, configs):
        super(SpectrogramModel, self).__init__()
        
        self.fc1 = nn.Linear(configs.joints * 2 * 2 * configs.input_freqs, 1024)
        self.fc2 = nn.Linear(1024, 512) 
        self.fc3 = nn.Linear(512, 28)      
        
    def forward(self, data):
        
        # data is Joints x (l/r) x (x,y) x freqs
        J, _, _, F = data.shape
        
        data = torch.flatten(data)
        
        layer_1 = nn.functional.relu(self.fc1(data))
        layer_2 = nn.functional.relu(self.fc2(layer_1))
        output = nn.functional.softmax(self.fc3(layer_2))
        
        return output
        
        
        

        
    