import torch
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
import pickle
from time import time

def main():
    """ 
        Loads the trained model and produces movie of the results.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirname = os.path.dirname(__file__)
    file_model = "ks_model_v3.pth"; "ks_model_v2.pth"; "ks_model.pth"; 
    pth_file = os.path.join(dirname, 'models', file_model)
    file = "ks_soln_ft_N_128_dt_0.25_tmax_1000.pt"
    filename = os.path.join(dirname, 'training_data', file)
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"File {pth_file} not found.")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    # Load the time series and segment it into smaller trajectories
    traj = torch.load(filename)[1:].numpy()
    traj_list, uscales = utils.segment_data(data=traj, nLengthTraj=20)
    info = utils.generate_info_dict(train_ratio=0.6, val_ratio=0.2, traj_list=traj_list, uscales=uscales)

    # Create the dataset and dataloader
    test_data = KSDataset.KSDataset(info=info, train_key="train", set_type="test")
    test_dataloader = DataLoader(test_data, batch_size=1)

    # Loss Function
    lam = 1e-2; 0
    loss_fn = KSLossFunc.KSL1RegRealDtMeanSquaredError(lam=lam)

    # Get the scales from the test_data
    test_scales = test_data.uscales
    scale = 10; 4.5

    # Load the model in evaluation mode
    N = 128
    model = KSGrayBox.KSGrayBox(h=0.25, N=N, uscales=uscales, return_coeffs=True).to(device)
    model.load_state_dict(torch.load(pth_file))
    model.eval()

    def test_loop(_dataloader, _model, _loss_fn):
        size = len(_dataloader.dataset)
        num_batches = len(_dataloader)
        test_loss = 0
        predictions = []
        predictions_dt = []
        truth = []
        truth_dt = []
        with torch.no_grad():
            times = torch.arange(_dataloader.dataset.useq.size(1)).type(torch.float32)
            for ii, (y0, Y) in enumerate(_dataloader):
                y, ydt = Y
                Y = torch.cat([y, ydt], dim=0)
                ydt = ydt.to(device)
                y = y.to(device)
                pred = _model(y0, steps=y.size(1))
                pred = torch.fft.ifft(pred, dim=-1).real
                pred_dt = NaturalCubicSpline(natural_cubic_spline_coeffs(times, pred)).derivative(times)

                # Rescale output for visualization
                predictions.append(pred.squeeze() * scale)
                predictions_dt.append(pred_dt.squeeze() * scale)
                truth.append(y.squeeze() * scale)
                truth_dt.append(ydt.squeeze() * scale)

                pred = torch.cat([pred, pred_dt], dim=0)
                coeffs = _model.return_coeffs()
                print(f"coeffs: {coeffs}")
                test_loss += _loss_fn(pred, Y, coeffs).item()

            predictions = torch.cat(predictions, dim=0)
            predictions_dt = torch.cat(predictions_dt, dim=0)
            truth = torch.cat(truth, dim=0)
            truth_dt = torch.cat(truth_dt, dim=0)
        test_loss /= num_batches
        return predictions, truth, predictions_dt, truth_dt, test_loss
    
    # Test the model
    predictions, truth, predictions_dt, truth_dt, test_loss = test_loop(test_dataloader, model, loss_fn)
    print(f"Test Loss: {test_loss}")

    # Animate the results
    # Print min/max of real space data
    print(f"Min of truth: {truth.min()}")
    print(f"Max of truth: {truth.max()}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Tests shape: {truth.shape}")

    # Animate the results
    x = 32*np.pi*np.arange(1,N+1)/N
    filename = 'spectral_kurasiv_1d_prediction_vs_truth_with_derivs'
    utils.animate_prediction_vs_truth(
        x=x, 
        predictions=predictions, 
        truth=truth,
        save=False, 
        filename=filename
        )
    
    utils.animate_prediction_vs_truth(
        x=x, 
        predictions=predictions_dt, 
        truth=truth_dt,
        save=False, 
        filename=filename
    )
    
    return

if __name__ == "__main__":
    main()

    






