import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
import sys
sys.path.append('../lib/')
import KSDataset
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
    
    info = utils.generate_info_dict(train_ratio=0.6, val_ratio=0.2, traj_list=traj_list, uscales=uscales)

    train_dataset = KSDataset.KSDataset(info=info, train_key="train", set_type="train")
    val_dataset = KSDataset.KSDataset(info=info, train_key="train", set_type="val")
    test_dataset = KSDataset.KSDataset(info=info, train_key="train", set_type="test")

    print(f"Number of training data: {len(train_dataset)}")
    print(f"Number of validation data: {len(val_dataset)}")
    print(f"Number of test data: {len(test_dataset)}")

    # Verify length of data
    assert len(train_dataset) == num_train, f"Expected {num_train} but got {len(train_dataset)}"
    assert len(val_dataset) == num_val, f"Expected {num_val} but got {len(val_dataset)}"
    assert len(test_dataset) == num_test, f"Expected {num_test} but got {len(test_dataset)}"

    # Check dimensions of data
    num = 0
    for x, y in train_dataset:
        print(f"Input data shape: {x.shape}")
        print(f"Output data shape: {y.shape}")
        num += 1
        if num == 1:
            break
    print(f"Number of data: {num}")

    return

if __name__ == "__main__":
    main()
