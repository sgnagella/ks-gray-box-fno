import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

class KSMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the mean squared error between the predicted
        and true values.
    """
    def __init__(self):
        super(KSMeanSquaredError, self).__init__()

    def forward(self, pred, y):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are complex-valued predicted Fourier coefficients and their true values.
            Error is computed as the sum of the squared real and imaginary parts, which is 
            equivalent to real-space error by Parseval-Plancherel theorem.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
        """
        err = pred - y
        return torch.mean(err.real**2 + err.imag**2)
    
class KSL2RegMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the mean squared error between the predicted
        and true values with a regularization on the learned coefficients.
    """
    def __init__(self, lam=1e-3):
        super(KSL2RegMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, pred, y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are complex-valued predicted Fourier coefficients and their true values.
            Error is computed as the sum of the squared real and imaginary parts, which is 
            equivalent to real-space error by Parseval-Plancherel theorem.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
        """
        err = pred - y
        return torch.mean(err.real**2 + err.imag**2) + self.lam * torch.sum(coeffs**2)
    
class KSL1RegMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the mean squared error between the predicted
        and true values with a regularization on the learned coefficients.
    """
    def __init__(self, lam=1e-3):
        super(KSL1RegMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, pred, y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are complex-valued predicted Fourier coefficients and their true values.
            Error is computed as the sum of the squared real and imaginary parts, which is 
            equivalent to real-space error by Parseval-Plancherel theorem.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
        """
        err = pred - y
        return torch.mean(err.real**2 + err.imag**2) + self.lam * torch.sum(torch.abs(coeffs))
    
class KSL1RegNNMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the mean squared error between the predicted
        and true values with a regularization on the learned coefficients.
    """
    def __init__(self, lam=1e-3):
        super(KSL1RegNNMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, pred, y, _model):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are complex-valued predicted Fourier coefficients and their true values.
            Error is computed as the sum of the squared real and imaginary parts, which is 
            equivalent to real-space error by Parseval-Plancherel theorem.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
                _model: torch.nn.Module, the model
        """
        err = pred - y
        eps = 1e-8
        return torch.mean(err.real**2 + err.imag**2) + self.lam * torch.mean(torch.sqrt(parameters_to_vector(_model.parameters())**2 + eps))