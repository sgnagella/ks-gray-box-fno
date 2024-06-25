import torch
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
import torch.optim as optim
import numpy as np
import sys
sys.path.append('util/')
sys.path.append('lib/')
from util import utils
from lib import KSDataset, KSGrayBox, KSLossFunc
import os
from time import time

def main():
    """ 
        Perform training of the gay box model to reproduce the Kuramoto-Sivashinsky
        dynamics.
        Get the parameters, training, and validation data 
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirname = os.path.dirname(__file__)
    file = "ks_soln_ft_N_128_dt_0.25_tmax_1000.pt"
    filename = os.path.join(dirname, 'training_data', file)
    dest_file = 'ks_model_v3.pth'; 'ks_model_v2.pth'; 'ks_model.pth'; 
    dest_name = os.path.join(dirname, 'models', dest_file)
    info_dest_name = os.path.join(dirname, 'models', 'ks_model_info.pickle')
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    os.makedirs(os.path.join(dirname, 'models'), exist_ok=True)

    # Load the time series and segment it into smaller trajectories

    traj = torch.load(filename)[1:].numpy()
    traj_list, uscales = utils.segment_data(data=traj, nLengthTraj=20)
    info = utils.generate_info_dict(train_ratio=0.6, val_ratio=0.2, traj_list=traj_list, uscales=uscales)

    # Create the dataset and dataloader
    train_data = KSDataset.KSDataset(info=info, train_key="train", set_type="train")
    val_data = KSDataset.KSDataset(info=info, train_key="train", set_type="val")
    test_data = KSDataset.KSDataset(info=info, train_key="train", set_type="test")

    train_dataloader = DataLoader(train_data, batch_size=5, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=5, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1)

    # Load the model 
    model = KSGrayBox.KSGrayBox(h=0.25, N=128, uscales=uscales, return_coeffs=True).to(device)
    lr = 1e-3; 1e-4
    betas =  (0.9, 0.7); (0.9, 0.999)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-7, weight_decay=0, amsgrad=True)
    lam = 1e-2; 1e-1; 0
    loss_fn = KSLossFunc.KSL2RegRealDtCorrWeightMeanSquaredError(lam=lam)

    # exit()
    def train_loop(_dataloader, _model, _loss_fn, _optimizer):
        size = len(_dataloader.dataset)
        times = torch.arange(_dataloader.dataset.useq.size(1)).type(torch.float32)
        for y0, Y in _dataloader:
            # Compute prediction and loss
            y, ydt = Y
            Y = torch.cat([y, ydt], dim=0)
            ydt = ydt.to(device)
            y = y.to(device)
            pred = _model(y0, steps=y.size(1))
            pred = torch.fft.ifft(pred, dim=-1).real
            pred_dt = NaturalCubicSpline(natural_cubic_spline_coeffs(times, pred)).derivative(times)
            pred = torch.cat([pred, pred_dt], dim=0)
            
            loss = loss_fn(pred, Y, _model.return_coeffs())

            # Backpropagation
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

        return

    def val_loop(_dataloader, _model, _loss_fn):
        size = len(_dataloader.dataset)
        num_batches = len(_dataloader)
        val_loss = 0
        times = torch.arange(_dataloader.dataset.useq.size(1)).type(torch.float32)
        with torch.no_grad():
            for y0, Y in _dataloader:
                y, ydt = Y
                Y = torch.cat([y, ydt], dim=0)
                ydt = ydt.to(device)
                y = y.to(device)
                pred = _model(y0, steps=y.size(1))
                pred = torch.fft.ifft(pred, dim=-1).real
                pred_dt = NaturalCubicSpline(natural_cubic_spline_coeffs(times, pred)).derivative(times)
                pred = torch.cat([pred, pred_dt], dim=0)

                val_loss += _loss_fn(pred, Y, _model.return_coeffs()).item()
        val_loss /= num_batches
        print(f"val_loss: {val_loss:.3e}")
        return val_loss
    
    def test_loop(_dataloader, _model, _loss_fn):
        size = len(_dataloader.dataset)
        num_batches = len(_dataloader)
        test_loss = 0
        times = torch.arange(_dataloader.dataset.useq.size(1)).type(torch.float32)
        with torch.no_grad():
            for y0, Y in _dataloader:
                y, ydt = Y
                Y = torch.cat([y, ydt], dim=0)
                y = y.to(device)
                pred = _model(y0, steps=y.size(1))
                pred = torch.fft.ifft(pred, dim=-1).real
                pred_dt = NaturalCubicSpline(natural_cubic_spline_coeffs(times, pred)).derivative(times)
                pred = torch.cat([pred, pred_dt], dim=0)

                test_loss += _loss_fn(pred, Y, _model.return_coeffs()).item()
        test_loss /= num_batches
        return pred, test_loss
    
    EPOCHS = 2000
    PATIENCE = 150
    counter = 0
    best_loss = np.inf
    checkpoint = True # continues training from the last checkpoint
    
    try:
        if os.path.isfile(dest_name) and checkpoint:
            model.load_state_dict(torch.load(dest_name))
            print("Model loaded to continue training.")

        toc = time()
        for t in range(EPOCHS):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            val_loss = val_loop(val_dataloader, model, loss_fn)
            counter += 1
            if val_loss < best_loss:
                best_loss = val_loss
                best_dict = deepcopy(model.state_dict())
                torch.save(best_dict, dest_name)
                print(f'best val_loss: {val_loss}')
                counter = 0
            if counter > PATIENCE:
                print("ran out of patience")
                break
        tic = time()

        model.load_state_dict(torch.load(dest_name))
        model.eval()
        test_predictions, test_loss = test_loop(test_dataloader, model, loss_fn)
        train_time = tic - toc
        info = dict(test_predictions=test_predictions, test_loss=test_loss,
                    train_time=train_time, model_state_dict=model.state_dict())
        utils.export_dict(info, info_dest_name)
        print(f"Test Loss: {test_loss}")

    except KeyboardInterrupt:
        torch.save(best_dict, dest_name)
        print("Model saved.")

    return

if __name__ == "__main__":
    main()

    






