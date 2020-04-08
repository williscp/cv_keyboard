import torch 
import torch.nn as nn
import numpy as np

class SpectrogramModel(nn.Module):
    
    def __init__(self, configs):
        super(SpectrogramModel, self).__init__()
        
        self.fc1 = nn.Linear(configs.joints * 2 * 2 * configs.input_freqs, 1024)
        self.fc2 = nn.Linear(1024, 1024) 
        self.fc3 = nn.Linear(1024, 1024) 
        self.fc4 = nn.Linear(1024, 512) 
        self.fc5 = nn.Linear(512, 28)   
        
        self.batch1 = nn.BatchNorm1d(1024)
        self.batch2 = nn.BatchNorm1d(1024)
        self.batch3 = nn.BatchNorm1d(1024)
        self.batch4 = nn.BatchNorm1d(512)
        
    def forward(self, data):
        
        # data is Joints x (l/r) x (x,y) x freqs
        B, J, _, _, F, T = data.shape
        
        data = data.reshape(B, J * 2 * 2 * F, T)
        data = data.permute(2, 0, 1)
        data = data.reshape(B * T, J * 2 * 2 * F)
        
        layer_1 = nn.functional.relu(self.batch1(self.fc1(data)))
        layer_2 = nn.functional.relu(self.batch2(self.fc2(layer_1)))
        layer_3 = nn.functional.relu(self.batch3(self.fc3(layer_2)))
        layer_4 = nn.functional.relu(self.batch4(self.fc4(layer_3)))
        output = nn.functional.softmax(self.fc5(layer_4))
        
        output = output.reshape(T, B, -1)
        output = output.permute(1, 2, 0)
        
        return output
    
class SpectrogramConvModel(nn.Module):
    
    def __init__(self, configs):
        super(SpectrogramConvModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=21 * 2 * 2, out_channels=256, kernel_size=(3,3), stride=1, padding=(0,1))
        self.batch1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=1, padding=(0,0))
        self.batch2 = nn.BatchNorm2d(num_features=512)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,1), stride=None, padding=0)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=(0,1)) 
        self.batch3 = nn.BatchNorm2d(num_features=512)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5,1), stride=(1,1), padding=0, output_padding=0)
        self.batch4 = nn.BatchNorm2d(num_features=256)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,1), stride=(2,1), padding=0, output_padding=0)
        self.batch5 = nn.BatchNorm2d(num_features=128)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3,1), stride=(2,1), padding=0, 
output_padding=0)
        self.batch6 = nn.BatchNorm2d(num_features=64)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,1), stride=1, padding=0, output_padding=0)
        self.batch7 = nn.BatchNorm2d(num_features=32)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4,1), stride=1, padding=0, output_padding=0)
        self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1,1), stride=1, padding=0, output_padding=0)

    
    def forward(self, data):
        
        #data is J x (l/r) x (x,y) x freqs x times 
        # need to be reshaped to 1 x (J * 2 * 2) x freqs x times
        
        B, J, _, _, F, T = data.shape
        
        data = data.reshape(B, J * 2 * 2, F, T)
        
        layer_1 = nn.functional.relu(self.batch1(self.conv1(data)))
        layer_2 = nn.functional.relu(self.batch2(self.conv2(layer_1)))
        layer_2_pooled = self.pool1(layer_2)
        layer_3 = nn.functional.relu(self.batch3(self.conv3(layer_2_pooled)))
        
        #print(layer_3.shape)
        layer_4 = nn.functional.relu(self.batch4(self.deconv1(layer_3)))
        layer_5 = nn.functional.relu(self.batch5(self.deconv2(layer_4)))
        layer_6 = nn.functional.relu(self.batch6(self.deconv3(layer_5)))
        layer_7 = nn.functional.relu(self.batch7(self.deconv4(layer_6)))
        
        layer_8 = self.deconv6(self.deconv5(layer_7))
        
        #print(layer_7.shape)
        layer_8 = layer_8.squeeze(0)
        #output = nn.functional.log_softmax(layer_8, dim=1)
        
        return layer_8
        
        
        
        

        
    