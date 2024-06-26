import torch
import torch.nn as nn
import sys
from torch.nn.utils import parameters_to_vector
# sys.path.append('../util/')
# from util import compute_auto_correlation

def compute_auto_correlation(*, data):
    """
        Computes the autocorrelation in the fourier coefficients of the data.
        data: torch tensor of shape (Nbatch, tspan, Nmodes)
        out: torch tensor of shape (Nbatch, tspan)
    """

    Nbatch = data.size(0)
    tspan = data.size(1)
    autocorr = torch.zeros((Nbatch, tspan))
    shifts = torch.arange(tspan)

    for ii, shift in enumerate(shifts): 
        diffs = data[..., :-shift if shift else None] - data[..., shift:]
        autocorr[..., ii] = torch.sum(torch.mean(torch.real(diffs*torch.conj(diffs)), dim=-1), dim=-1)

    return autocorr

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
    
class KSL1RegRealMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the real-space mean squared error between the predicted
        and true values with a regularization on the learned coefficients.
    """
    def __init__(self, lam=1e-3):
        super(KSL1RegRealMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, pred, y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are real-valued predicted field and their true values.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
        """
        err = pred - y
        eps = 1e-8
        return torch.mean(err**2) + self.lam * torch.mean(torch.sqrt(coeffs**2 + eps))
    

class KSL1RegRealDtMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the real-space mean squared error between the predicted
        and true values with a regularization on the learned coefficients, also 
        including time derivatives.
    """
    def __init__(self, lam=1e-3):
        super(KSL1RegRealDtMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, Pred, Y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are real-valued predicted field and their true values.

            Inputs:
                Pred: torch.Tensor (2*batch_size, tspan, Nmodes), predicted solution and its time derivative
                y: torch.Tensor (2*batch_size, tspan, Nmodes), true solution and its time derivative 
        """
        batch_size = Pred.size(0)//2
        err = Pred - Y
        relative_weight  = 0.5
        weight = relative_weight * torch.ones(Pred.size(0))
        time_deriv_loss_weight = (1 - relative_weight)*Pred.size(1) # also multiply by number of time points to make it comparable to the real space loss
        weight[batch_size:] *= time_deriv_loss_weight
        weight = weight[:, None, None] 
        # err[batch_size:] = err[batch_size:] * weight
        eps = 1e-10
        return torch.mean(weight * err**2) + self.lam * torch.mean(torch.sqrt(coeffs**2 + eps))

class KSL2RegRealDtMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the real-space mean squared error between the predicted
        and true values with a regularization on the learned coefficients, also 
        including time derivatives.
    """
    def __init__(self, lam=1e-3):
        super(KSL2RegRealDtMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, Pred, Y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are real-valued predicted field and their true values.

            Inputs:
                Pred: torch.Tensor (2*batch_size, tspan, Nmodes), predicted solution and its time derivative
                y: torch.Tensor (2*batch_size, tspan, Nmodes), true solution and its time derivative 
        """
        batch_size = Pred.size(0)//2
        err = Pred - Y
        weight = torch.ones(Pred.size(0))
        time_deriv_loss_weight = 10
        weight[batch_size:] *= time_deriv_loss_weight
        weight = weight[:, None, None] 
        # err[batch_size:] = err[batch_size:] * weight
        eps = 1e-10
        return torch.mean(weight * err**2) + self.lam * torch.mean(coeffs**2)

class KSL2RegRealDtCorrWeightMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the real-space mean squared error between the predicted
        and true values with a regularization on the learned coefficients, also 
        including time derivatives.
        Now this has exponential weighting depending on the time autocorrelation of the data.
    """
    def __init__(self, lam=1e-3):
        super(KSL2RegRealDtCorrWeightMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, Pred, Y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are real-valued predicted field and their true values.

            Inputs:
                Pred: torch.Tensor (2*batch_size, tspan, Nmodes), predicted solution and its time derivative
                y: torch.Tensor (2*batch_size, tspan, Nmodes), true solution and its time derivative 
        """
        batch_size = Pred.size(0)//2
        tspan = Pred.size(1)
        err = (Pred - Y)
        corr_weight = 10
        steepness = 0.1
        # weight = 1 - torch.exp( -(corr_weight * 1/compute_auto_correlation(data=err[:batch_size])) )
        # weight = corr_weight * torch.exp( - ( steepness * compute_auto_correlation(data=err[:batch_size]) + 0.1 )**-1 )
        weight = corr_weight * torch.ones((batch_size, tspan))
        weight = torch.cat([torch.ones((batch_size, tspan)), weight], dim=0)
        # print(weight)
        # err[batch_size:] = err[batch_size:] * weight
        eps = 1e-10
        return torch.mean(weight[..., None] * err**2) + self.lam * torch.mean(coeffs**2)      

class KSL2RegRealMeanSquaredError(nn.Module):
    """
        Loss function for the Kuramoto-Sivashinsky gray-box model.
        The loss function is the real-space mean squared error between the predicted
        and true values with a regularization on the learned coefficients.
    """
    def __init__(self, lam=1e-3):
        super(KSL2RegRealMeanSquaredError, self).__init__()
        self.lam = lam

    def forward(self, pred, y, coeffs):
        """
            Forward pass of the loss function.
            Computes the mean squared error between the predicted and true values.
            Inputs are real-valued predicted field and their true values.

            Inputs:
                pred: torch.Tensor (batch_size, tspan, Nmodes), predicted Fourier coefficients
                y: torch.Tensor (batch_size, tspan, Nmodes), true Fourier coefficients
        """
        err = pred - y
        eps = 1e-8
        return torch.mean(err**2) + self.lam * (torch.mean(coeffs ** 2))
    
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
        eps = 1e-8
        return torch.mean(err.real**2 + err.imag**2) + self.lam * torch.mean(torch.sqrt(coeffs**2 + eps))
    
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