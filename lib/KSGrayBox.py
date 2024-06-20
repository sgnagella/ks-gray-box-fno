import torch
import torch.nn as nn
from torch.fft import fft, ifft
import numpy as np

class MLP(nn.Module):
    """
        Class to define the neural network architecture for the ODE function.
    """
    def __init__(self, nn_dims):
        super().__init__()
        layers = []
        for ind in range(len(nn_dims) - 1):
            if ind != 0:
                layers.append(nn.ReLU())
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
    def __init__(self, N, return_coeffs=False):
        super(ODEMLPFunc, self).__init__()
        hidden = 16
        output = 4
        n_embeddings = 4
        self.mlp = MLP([n_embeddings*N, hidden, output])
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
        self.aold = None
        self.aold1 = None
        self.aold2 = None

        self.bold = None
        self.bold1 = None
        self.bold2 = None

        self.cold = None
        self.cold1 = None
        self.cold2 = None

        return

    def return_feature_matrix(self,x): 
        # Takes input x in fourier space and returns real space feature matrix
        # feature matrix is current state and its past 
        xreal = ifft(x, dim=-1).real

        # print(f"in KSGraybox.py return_feature_matrix: xreal.size() = {xreal.size()}")
        x = torch.stack([torch.ones_like(x), xreal, xreal**2, xreal**3]).type(torch.float32)

        x = torch.permute(x, (1,2,0,3)) # size(batch_size, 1, 2, N)
        # print(f"in KSGraybox.py return_feature_matrix: x.size() = {x.size()}")

        return x
    
    def return_x_input(self, x, xold, xold1, xold2): 
        out = ifft(torch.stack([x, xold, xold1, xold2]), dim=-1).real
        out = torch.permute(out, (1,2,0,3))
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print(f"in KSGraybox.py return_x_input: out.size() = {out.size()}")
        return out

    def forward(self, x, xold, xold1, xold2):
        # Inputs to model are current state and past state in Fourier space
        Nv = self.g * fft(self.odefunc(0, self.return_x_input(x, xold, xold1, xold2), self.return_feature_matrix(x)), dim=-1).type(torch.complex64)
        
        # self.aold = xold
        # self.aold1 = xold1

        a = self.E2 * x + self.Q *  Nv
        Na = self.g * fft(self.odefunc(0, self.return_x_input(a, self.aold, self.aold1, self.aold2), self.return_feature_matrix(a)), dim=-1).type(torch.complex64)
        self.aold2 = self.aold1.clone()
        self.aold1 = self.aold.clone()
        self.aold = a.clone()

        # self.bold = self.aold
        # self.bold1 = self.aold1

        b = self.E2 * x + self.Q * Na
        Nb = self.g * fft(self.odefunc(0, self.return_x_input(b, self.bold, self.bold1, self.bold2), self.return_feature_matrix(b)), dim=-1).type(torch.complex64)
        self.bold2 = self.bold1.clone()
        self.bold1 = self.bold.clone()
        self.bold = b.clone()

        # self.cold = self.bold
        # self.cold1 = self.bold1
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self.g * fft(self.odefunc(0, self.return_x_input(c, self.cold, self.cold1, self.cold2), self.return_feature_matrix(c)), dim=-1).type(torch.complex64)
        self.cold2 = self.cold1.clone()
        self.cold1 = self.cold.clone()
        self.cold = c.clone()

        x1 = self.E * x + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        return x1
    

class MultiStep(nn.Module):
    """
        Wrapper class for the SingleStep class to apply the single step multiple times.    
    """
    def __init__(self, N,h, uscales, return_coeffs=False):
        """
            Initialize the MultiStep class.
            Inputs:
                h: float, step size for the single step
                N: int, number of Fourier modes
                uscales: scales for the Fourier modes
                return_coeffs: bool, whether to return the coefficients
        """
        super(MultiStep, self).__init__()
        odefunc = ODEMLPFunc(N, return_coeffs=return_coeffs)
        self.stepper = SingleStep(odefunc, N,h)
    
    def forward(self, x, steps):
        xs = []
        # Construct vector of current and past states in Fourier space
        xold = x.clone()
        xold1 = x.clone()
        xold2 = x.clone()

        self.stepper.aold = xold
        self.stepper.aold1 = xold1
        self.stepper.aold2 = xold2

        self.stepper.bold = xold
        self.stepper.bold1 = xold1
        self.stepper.bold2 = xold2

        self.stepper.cold = xold
        self.stepper.cold1 = xold1
        self.stepper.cold2 = xold2

        for step in range(steps):
            t = self.stepper.h*step
            xp = self.stepper(x, xold, xold1, xold2)
            xold2 = xold1.clone()
            xold1  = xold.clone()
            xold = x.clone()
            x = xp
            if step % int(1/self.stepper.h) == 0:
                xs.append(x)

        # Concatenate along the time axis (dim=1)
        return torch.cat(xs, dim=1)
    
class KSGrayBox(nn.Module): 
    """
        Wrapper class for the MultiStep module
    """

    def __init__(self, N,h, uscales, return_coeffs=False):
        """
            Initialize the KSGrayBox class.
            Inputs:
                h: float, step size for the single step
                N: int, number of Fourier modes
                uscales: scales for the Fourier modes
        """
        super(KSGrayBox, self).__init__()
        self.model = MultiStep(N,h, uscales, return_coeffs=return_coeffs)

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