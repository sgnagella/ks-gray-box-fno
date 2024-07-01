import torch
import torch.nn as nn
from torch.fft import fft, ifft
from LegPoly import * 
import numpy as np

class Spline_Activation(nn.Module):
    def __init__(self):
        super(Spline_Activation, self).__init__()

    def forward(self, x):
        zeros = torch.zeros(x.size())
        out1 = torch.min(torch.max(zeros-1,x) , zeros)
        out2 = torch.min(-torch.min(zeros+1, x), zeros) 
        return 1 + out1 + out2

class MLP(nn.Module):
    """
        Class to define the neural network architecture for the ODE function.
    """
    def __init__(self, nn_dims):
        super().__init__()
        layers = []
        for ind in range(len(nn_dims) - 1):
            if ind != 0:
                # layers.append(nn.ReLU())
                layers.append(Spline_Activation())
            layers.append(nn.Linear(nn_dims[ind], nn_dims[ind + 1],
                                    dtype=torch.float32,
                                    bias=False))   
        self.mlp = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def forward(self, x):
        output = self.mlp(x)
        return output

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, 0, 0.5).float()

        return

class ODEMLPFunc(nn.Module):
    """
        Class to define the ODE function in terms of the neural network (the MLP)
    """
    def __init__(self, n_modes, n_embeddings=6, return_coeffs=False):
        super(ODEMLPFunc, self).__init__()
        hidden = 8
        output = 4
        self.n_modes = n_modes
        self.mlp = MLP([n_embeddings*n_modes, hidden, output])
        self.return_coeffs = return_coeffs
        self.coeffs = None
     
    def forward(self, t, xinput, xfeature):
        # Learning the "time"-dependent coefficients 
        # Instead of solving for coefficients in Ax = b style, 
        # use a neural network to learn the coefficients given 
        # a sufficient basis, which is the feature matrix, x
        # Inputs are the current state and its past values
        
        # return self.mlp(x).squeeze(-1) @ x

        # input to bmm is (batch_size, 1, 4, 1) (output of NN)
        # 2nd input to bmm is feature matrix (batch_size, 1, 4, N)
        # returns shape (batch_size, 1, N)
        if self.return_coeffs:
            self.coeffs = self.mlp(xinput)
            return torch.bmm(self.coeffs, xfeature.squeeze(1))
        return torch.bmm(self.mlp(xinput), xfeature.squeeze(1))

# Stepping procedure for the RNN
class SingleStep(nn.Module):
    """
        Class to define the single step of the RNN (ETD RK4 method for solving KS equation)
    """
    def __init__(self, odefunc, N, h):
        super(SingleStep, self).__init__()
        self.odefunc = odefunc

        # Pre-compute quantities for the ETD RK4 method
        k = torch.cat([torch.arange(0,N/2),torch.tensor([0.]),torch.arange(-N/2+1,0)],0)/16
        k = k.detach()
        filter = torch.abs(k) < (1/3) * N/2

        g = -.5j*k
        L = k**2 - k**4 
        E = (h*L).exp()
        E2 = (h*L/2).exp()
        M = 16
        r = (1j*torch.pi*(torch.arange(1,M+1)-.5)/M)
        r = r.type(torch.complex64)
        LR = h*L[:,None].repeat_interleave(M,1) + r[None,:].repeat_interleave(N,0)
        Q = h*(((LR/2).exp()-1)/LR).mean(dim=1).real
        f1 = h*((-4-LR+LR.exp()*(4-3*LR+LR**2))/LR**3).mean(dim=1).real
        f2 = h*((2+LR+LR.exp()*(-2+LR))/LR**3).mean(dim=1).real
        f3 = h*((-4-3*LR-LR**2+LR.exp()*(4-LR))/LR**3).mean(dim=1).real

        self.h = h
        self.k = k
        self.g = g
        self.L = L
        self.E = E
        self.E2 = E2
        self.M = M
        self.r = r
        self.LR = LR
        self.Q = Q
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.filter = filter

        # Store the old values of intermediate solutions
        self.xold = None
        self.aold = None
        self.bold = None
        self.cold = None

        return

    def update_xold(self, xold, xnew): 
        for ii in range(len(xold)-1): 
            xold[ii] = xold[ii+1].clone()
        xold[-1] = xnew.clone()
        return xold

    def return_feature_matrix(self,x): 
        # Takes input x in fourier space and returns real space feature matrix
        # feature matrix is current state and its past 
        xreal = ifft(x, dim=-1).real

        # print(f"in KSGraybox.py return_feature_matrix: xreal.size() = {xreal.size()}")
        # x = torch.stack([torch.ones_like(x), xreal, xreal**2, xreal**3]).type(torch.float32)
        # x = torch.stack([xreal, xreal**2, xreal**3, xreal**4]).type(torch.float32)
        # x = torch.stack([LegendrePolynomial0.apply(xreal), LegendrePolynomial2.apply(xreal)]).type(torch.float32)
        x = torch.stack([
            LegendrePolynomial0.apply(xreal),
            LegendrePolynomial1.apply(xreal),
            LegendrePolynomial2.apply(xreal), 
            # LegendrePolynomial4.apply(xreal),
            LegendrePolynomial3.apply(xreal)
            ]).type(torch.float32)
        
        # print(f"in KSGraybox.py return_feature_matrix: x.max = {torch.max(x)}, x.min = {torch.min(x)}")

        # x = torch.stack([LegendrePolynomial0.apply(xreal), LegendrePolynomial2.apply(xreal)]).type(torch.float32)

        x = torch.permute(x, (1,2,0,3)) # size(batch_size, 1, 2, N)
        # print(f"in KSGraybox.py return_feature_matrix: x.size() = {x.size()}")

        return x
    
    def return_x_input(self, xold): 
        out = ifft(torch.stack(xold), dim=-1).real
        out = torch.stack(xold)
        out = torch.permute(out, (1,2,0,3))

        # Compute the SVD of the input data on the last two dimensions
        U, S, Vh = torch.linalg.svd(out, full_matrices=False)
        Vh = Vh.mH[..., :self.odefunc.n_modes]              # size(batch_size, 1, N, n_modes)
        out = torch.matmul(out, Vh)                         # size(batch_size, time, n_embeddings, n_modes)
        out = out.reshape(out.shape[0], out.shape[1], -1)   # size(batch_size, time, n_embeddings*n_modes)
        # print(f"in KSGraybox.py return_x_input: out.size() = {out.size()}")
        return out

    def nonlinear(self, xold, x): 
        return self.g * fft(self.odefunc(0, self.return_x_input(xold), self.return_feature_matrix(x)), dim=-1).type(torch.complex64)

    def forward(self, x):
        # Inputs to model are current state and past state in Fourier space
        Nv = self.nonlinear(self.xold, x)

        a = self.E2 * x + self.Q *  Nv
        self.aold = self.update_xold(self.aold, a)
        Na = self.nonlinear(self.aold, a)

        b = self.E2 * x + self.Q * Na
        self.bold = self.update_xold(self.bold, b)
        Nb = self.nonlinear(self.bold, b)

        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        self.cold = self.update_xold(self.cold, c)
        Nc = self.nonlinear(self.cold, c)

        x = self.E * x + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        self.xold = self.update_xold(self.xold, x)
        return x
    

class MultiStep(nn.Module):
    """
        Wrapper class for the SingleStep class to apply the single step multiple times.    
    """
    def __init__(self, N,h, uscales, n_embeddings=6, n_modes=5, return_coeffs=False):
        """
            Initialize the MultiStep class.
            Inputs:
                h: float, step size for the single step
                N: int, number of Fourier modes
                uscales: scales for the Fourier modes
                return_coeffs: bool, whether to return the coefficients
        """
        super(MultiStep, self).__init__()
        odefunc = ODEMLPFunc(N, n_modes, n_embeddings=n_embeddings, return_coeffs=return_coeffs)
        self.stepper = SingleStep(odefunc, N,h)
        self.n_embeddings = n_embeddings
    
    def forward(self, x, steps):
        xs = []
        # Construct vector of current and past states in Fourier space
        xold = [x.clone()]*(self.n_embeddings)

        self.stepper.xold = xold
        self.stepper.aold = xold
        self.stepper.bold = xold
        self.stepper.cold = xold

        for step in range(steps):
            t = self.stepper.h*step
            x = self.stepper(x)
            if step % int(1/self.stepper.h) == 0:
                xs.append(x)

        # Concatenate along the time axis (dim=1)
        return torch.cat(xs, dim=1)
    
class KSGrayBox(nn.Module): 
    """
        Wrapper class for the MultiStep module
    """

    def __init__(self, N,h, uscales, n_embeddings=6, n_modes=5, return_coeffs=False):
        """
            Initialize the KSGrayBox class.
            Inputs:
                h: float, step size for the single step
                N: int, number of Fourier modes
                uscales: scales for the Fourier modes
        """
        super(KSGrayBox, self).__init__()
        self.model = MultiStep(N,h, uscales, n_embeddings, n_modes, return_coeffs=return_coeffs)

    def return_coeffs(self): 
        return self.model.stepper.odefunc.coeffs

    def forward(self, y0, steps=1):
        """
            Forward pass of the model.
            Inputs:
                y0: torch.Tensor, initial condition
                steps: Number of time points to compute
            Outputs:
                pred: torch.Tensor, output data

            Given trajectory data, the model makes a prediction from the initial state.
        """

        assert y0.size(-1) == self.model.stepper.k.size(0),\
            f"Input data {y0.size(-1)} must have the same number of Fourier modes as the model {self.model.stepper.k.size(0)}"
        
        steps = int(steps / self.model.stepper.h)

        return self.model(y0,steps)