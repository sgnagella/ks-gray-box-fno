import torch
from torch.utils.data import Dataset
import numpy as np
import sys
sys.path.append('../util/')
import utils

class KSDataset(Dataset):
    """
        Dataset class for the Kuramoto-Sivashinsky data.
        Main driver for organizing the data for training and validation.
    """
    def __init__(self, info, train_key, set_type):
        # Load the complete training data and their scales. 
        training_data = info['training_data'][train_key]
        uscales = info['uscales'][train_key]
        n_train_traj, n_val_traj = info['numTrainTraj'][train_key]

        # Split into training, validation, and test data.
        (train_data, val_data, test_data) = utils.get_train_val_data(data_list=training_data,
                                                                     uscales=uscales,
                                                                     nTrainTraj=n_train_traj,
                                                                     nTrainValTraj=n_val_traj)
        if 'train' in set_type:
            data = train_data
        elif 'val' in set_type:
            data = val_data
        elif 'test' in set_type:
            data = test_data
        else:
            raise ValueError('set_type must be one of the following: ["train", "val", "test"')

        self.useq = torch.from_numpy(data['useq'])
        self.useq0 = torch.from_numpy(data['useq0'])
        self.uscales = {'umin': torch.from_numpy(data['uscales']['umin']).type(torch.complex64), 
                        'umax': torch.from_numpy(data['uscales']['umax']).type(torch.complex64)}
        # self.useq = torch.from_numpy(np.transpose(data['useq'], axes=(0, 2, 1)))

    def __len__(self):
        return self.useq.shape[0]

    def __getitem__(self, idx):
        return self.useq0[idx], self.useq[idx]