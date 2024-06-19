import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
from lib import KSDataset
import os

def main(): 
    # Load the data
    fn = '../training_data/ks_soln_ft_N_128_dt_0.25_tmax_500.pt'
    traj = torch.load(fn)[1:].numpy()
    nLengthTraj = 10

    # Segment the trajectory
    traj_list, uscales = utils.segment_data(data=traj, nLengthTraj=nLengthTraj)
    nTraj = len(traj_list)
    print(f"Number of trajectories: {nTraj}")

    # Get the training and validation data
    train_ratio = 0.6
    val_ratio = 0.2

    num_train = int(train_ratio * nTraj)
    num_val = int(val_ratio * nTraj)
    num_test = nTraj - num_train - num_val

    train_data, val_data, test_data = utils.get_train_val_data(data_list=traj_list, 
                                                               uscales=uscales, 
                                                               nTrainTraj=num_train, 
                                                               nTrainValTraj=num_val)
    
    # Verify length of data
    assert len(train_data['useq']) == num_train, f"Expected {num_train} but got {len(train_data)}"
    assert len(val_data['useq']) == num_val, f"Expected {num_val} but got {len(val_data)}"
    assert len(test_data['useq']) == num_test, f"Expected {num_test} but got {len(test_data)}"

    return

if __name__ == "__main__":
    main()
