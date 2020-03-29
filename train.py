import torch
import os
import numpy as np

from config import Config 
from dataset import Dataset

configs = Config 

train_set = Dataset(configs)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, num_workers=0, pin_memory=False, shuffle=True, drop_last=True)



