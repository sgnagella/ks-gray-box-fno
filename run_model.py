import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
    pth_file = os.path.join(dirname, 'models', 'ks_model.pth')
    file = "ks_soln_ft_N_128_dt_0.25_tmax_500.pt"
    filename = os.path.join(dirname, 'training_data', file)
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"File {pth_file} not found.")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    # Load the time series and segment it into smaller trajectories
    # with torch.no_grad():
    #     traj = torch.fft.ifft(torch.load(filename)[1:], dim=-1).real.numpy()
    traj = torch.load(filename)[1:].numpy()
    traj_list, uscales = utils.segment_data(data=traj, nLengthTraj=20)
    info = utils.generate_info_dict(train_ratio=0.6, val_ratio=0.2, traj_list=traj_list, uscales=uscales)

    # Create the dataset and dataloader
    test_data = KSDataset.KSDataset(info=info, train_key="train", set_type="test")
    test_dataloader = DataLoader(test_data, batch_size=1)

    # Loss Function
    loss_fn = KSLossFunc.KSL2RegRealMeanSquaredError(lam=0)

    # Get the scales from the test_data
    test_scales = test_data.uscales
    scale = 10

    # Load the model in evaluation mode
    N = 128
    model = KSGrayBox.KSGrayBox(h=0.25, N=N, uscales=uscales, return_coeffs=True).to(device)
    model.load_state_dict(torch.load(pth_file))
    model.eval()

    def test_loop(_dataloader, _model, _loss_fn):
        size = len(_dataloader.dataset)
        # print(size)
        num_batches = len(_dataloader)
        # print(num_batches)
        test_loss = 0
        predictions = []
        truth = []
        with torch.no_grad():
            for ii, (y0, y) in enumerate(_dataloader):
                # y0 = torch.fft.fft(y0, dim=-1)
                y = y.to(device)
                pred = _model(y0, steps=y.size(1))

                y = torch.fft.ifft(y, dim=-1).real
                pred = torch.fft.ifft(pred, dim=-1).real
                coeffs = _model.return_coeffs()
                print(coeffs)

                test_loss += _loss_fn(pred, y, coeffs).item()

                # Rescale output for visualization
                pred = pred.squeeze() # Remove the batch dimension (only 1 batch size)
                # predictions.append(pred * (test_scales['umax'][ii] - test_scales['umin'][ii]) + test_scales['umin'][ii])
                # predictions.append(0.5*(pred+1) * (test_scales['umax'][ii] - test_scales['umin'][ii]) + test_scales['umin'][ii])
                # truth.append(y.squeeze() * (test_scales['umax'][ii] - test_scales['umin'][ii]) + test_scales['umin'][ii])
                # truth.append(0.5*(y.squeeze()+1) * (test_scales['umax'][ii] - test_scales['umin'][ii]) + test_scales['umin'][ii])

                predictions.append(pred * scale)
                truth.append(y.squeeze() * scale)

            predictions = torch.cat(predictions, dim=0)
            truth = torch.cat(truth, dim=0)
        test_loss /= num_batches
        return predictions, truth, test_loss
    
    # Test the model
    predictions, truth, test_loss = test_loop(test_dataloader, model, loss_fn)
    print(f"Test Loss: {test_loss}")

    # Animate the results
    # predictions = torch.fft.ifft(predictions, dim=1).detach().numpy()
    # truth = torch.fft.ifft(truth, dim=1).detach().numpy()

    # Print min/max of real space data
    print(f"Min of truth: {truth.min()}")
    print(f"Max of truth: {truth.max()}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Tests shape: {truth.shape}")

    # Animate the results
    x = 32*np.pi*np.arange(1,N+1)/N
    filename = 'spectral_kurasiv_1d_prediction_vs_truth'
    utils.animate_prediction_vs_truth(x=x, predictions=predictions, truth=truth, save=False, filename=filename)
    return

if __name__ == "__main__":
    main()

    






