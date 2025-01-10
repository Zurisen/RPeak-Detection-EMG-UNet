import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_ECG(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.device = device
        
        self.data_paths_X = glob.glob(self.data_dir + '/' + '*X_data*.pt')
        self.data_paths_y = glob.glob(self.data_dir + '/' + '*y_data*.pt')

    def __len__(self):
        return len(self.data_paths_X)

    def __getitem__(self, idx):
        X_data = torch.tensor(torch.load(self.data_paths_X[idx]), dtype=torch.float32).view(1,-1).to(self.device)
        y_data = torch.tensor(torch.load(self.data_paths_y[idx]), dtype=torch.float32).view(1,-1).to(self.device)
        return X_data, y_data
    
