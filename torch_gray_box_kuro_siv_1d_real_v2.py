#%%
"""
Define and train the gray box model to learn the Kuramoto-Sivashinsky equation

The 1D KS equations are defined as:
    ut + uxx + uxxxx + 0.5 * (ux)^2 = 0

The gray box model is defined as:
    ut + uxx + uxxxx + NN(u) = 0
    NN(u) = c0u + c2ux + c3u**2 + c4u*ux + c5ux**2

We want to 'discover' the underlying dynamics of the system

In this version, the input to the NN is the feature matrix of the solution in real space.
The NN will output the coefficient vector for the polynomial terms in the gray box model.
"""

#TODO: Compute solutions for initial conditions and train the model 
#TODO: Implement sparse regression in the loss function? minimize the number/magnitude of parameters
#TODO: Implement K-Fold cross validation to maximize the generalization of the model
#TODO: Divide the trajectory into smaller sub-trajectories for training -- implement with K-Fold cross validation
#TODO: Feature space is only the changing, oscillatory modes 
#TODO: Implement the model in PyTorch Lightning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

torch.set_anomaly_enabled(True)
#%% Define the model 

# Generate training data
N = 128
h = 0.25; 0.1
tmax = 500

x = 32*torch.pi*torch.arange(1,N+1)/N
u = torch.cos(x/16)*(1+torch.sin(x/16))
v = fft(u)

# Load the solution 
fn = f'ks_soln_ft_N_{N}_dt_{str(0.25)}_tmax_{tmax}'
vv = torch.load(fn + '.pt')[:-1]

# Divide by the k = 0 mode
# vv = vv / vv[0, N//2]

max_real_val = torch.max(ifft(vv).real)
min_real_val = torch.min(ifft(vv).real)

nmax = int(tmax/h)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

num_train = int(train_ratio * tmax)
num_val = int(val_ratio * tmax)
num_test = tmax - num_train - num_val

train_data = vv[:num_train]
val_data = vv[num_train:num_train + num_val]
test_data = vv[num_train + num_val:]

inputs = torch.tensor(train_data[0].flatten(), dtype=torch.complex64, requires_grad=True)
targets = torch.tensor(train_data[1:], dtype=torch.complex64)
val_i = torch.tensor(val_data[0].flatten(), dtype=torch.complex64) # input
val_o = torch.tensor(val_data[1:], dtype=torch.complex64)          # output
test_i = torch.tensor(test_data[0].flatten(), dtype=torch.complex64) # input
test_o = torch.tensor(test_data[1:], dtype=torch.complex64)          # output

#%%
# Initialize the neural network structure

# Define the MLP neural network
class MLP(nn.Module):
    def __init__(self, nn_dims):
        super().__init__()
        layers = []
        # for ind in range(len(nn_dims) - 1):
        #     if ind != 0 and ind != len(nn_dims) - 2:
        #         layers.append(nn.Sigmoid())
        #     else:
        #         layers.append(nn.ReLU())
        #     layers.append(nn.Linear(nn_dims[ind], nn_dims[ind + 1],
        #                             dtype=torch.float32,
        #                             bias=False))


        # for ind in range(len(nn_dims) - 1):
        #     if ind != 0 and ind > 2:
        #         layers.append(nn.ReLU())
        #     else: 
        #         layers.append(nn.Tanh())
        #     layers.append(nn.Linear(nn_dims[ind], nn_dims[ind + 1],
        #                             dtype=torch.float32,
        #                             bias=False))   
        # self.mlp = nn.Sequential(*layers)
        # self.apply(self.init_weights)


        for ind in range(len(nn_dims) - 1):
            if ind != 0:
                layers.append(nn.LeakyReLU())
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
            torch.nn.init.uniform_(m.weight, 0, 0.3).float()
            # torch.nn.init.eye_(m.weight).float()
            # torch.nn.init.normal_(m.weight, mean=0, std=0.01).float()
    

# Class to define the ODE function in terms of the neural network (the MLP)
class ODEMLPFunc(nn.Module):
    def __init__(self):
        super(ODEMLPFunc, self).__init__()
        # self.mlp = MLP([N//2 +1 , N//2 +1 , N//2 +1])
        size = 4
        self.mlp = MLP([N, size, size, 1])
        self.input = torch.ones(size)

    # def return_coeffs(self):
    #     return self.mlp(self.input)
     
    def forward(self, t, x):
        # Learning the "time"-dependent coefficients 
        # Instead of solving for coefficients in Ax = b style, 
        # use a neural network to learn the coefficients given 
        # a sufficient basis, which is the feature matrix, x
        return self.mlp(x).squeeze() @ x
    
# Stepping procedure for the RNN
class SingleStep(nn.Module):
    def __init__(self, odefunc, h):
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

        # Define operators to pick out the desired feature vector
        self.feature_op = torch.cat([torch.eye(N//2 +1), torch.zeros((N//2 -1, N//2 +1))], 0)
        self.feature_op = self.feature_op.type(torch.complex64)

        extraction_flip_op = torch.cat([torch.zeros((1, N//2 -1)) , torch.fliplr(torch.eye(N//2 -1)) , torch.zeros((1, N//2 -1))], 0)
        self.extraction_op = torch.cat([torch.eye(N//2 +1), extraction_flip_op], 1)
        self.extraction_op = self.extraction_op.type(torch.complex64)

    def return_feature_matrix(self,x): 
        # Takes input x in fouier space and outputs matrix of real-space polynomials
        xreal = ifft(x).real
        # dxreal = ifft(1j*self.k*x).real

        # Normalize 
        # xreal = (xreal - xreal.min()) / (xreal.max() - xreal.min())
        # print(xreal.isnan().sum())
        # dxreal = (dxreal - dxreal.min()) / (dxreal.max() - dxreal.min())

        # print(f"before:{x.size()}")
        # x = torch.stack([torch.ones_like(xreal), xreal*dxreal, xreal**2, dxreal**2]).type(torch.float32)
        x = torch.stack([torch.ones_like(x), xreal, xreal**2, xreal**3]).type(torch.float32)
        # print(f"after:{x.size()}")
        return x

    def forward(self, x):
        # Inputs to the NN are only the first N//2 -1 modes
        # Use symmetry to reconstruct the rest of the modes

        # iftx = ifft(x).real
        # print(((iftx - iftx.min()) / (iftx.max() - iftx.min())).detach().numpy())
        # Nv = fft(self.odefunc(0, (iftx - iftx.min()) / (iftx.max() - iftx.min())))
        # print('computing rk4')
        Nv = self.g * fft(self.odefunc(0, self.return_feature_matrix(x))).type(torch.complex64)
        a = self.E2 * x + self.Q *  Nv

        # ifta = ifft(a).real
        # Na = fft(self.odefunc(0, (ifta - ifta.min()) / (ifta.max() - ifta.min())))
        Na = self.g * fft(self.odefunc(0, self.return_feature_matrix(a))).type(torch.complex64)

        b = self.E2 * x + self.Q * Na

        # iftb = ifft(b).real
        # Nb = fft(self.odefunc(0, (iftb - iftb.min()) / (iftb.max() - iftb.min())))
        Nb = self.g * fft(self.odefunc(0, self.return_feature_matrix(b))).type(torch.complex64)

        c = self.E2 * a + self.Q * (2 * Nb - Nv)

        # iftc = ifft(c).real
        # Nc = fft(self.odefunc(0, (iftc - iftc.min()) / (iftc.max() - iftc.min())))
        Nc = self.g * fft(self.odefunc(0, self.return_feature_matrix(c))).type(torch.complex64)

        x1 = self.E * x + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        # print('done computing rk4')
        # print(x1.detach().numpy())
        # Zero out high frequency modes
        return x1

# A "recurrent" neural network which applies the single step multiple times
class MultiStep(nn.Module):
    def __init__(self, stepper):
        super(MultiStep, self).__init__()
        self.stepper = stepper
    
    def forward(self, x, steps):
        xs = []
        # x0 = x.clone().detach()[0]
        # xn2 = x.clone().detach()[N//2]
        for step in range(steps):
            t = h*step
            # print(x.size())
            x = self.stepper(x)
            # x[0] = x0
            # x[N//2] = xn2
            # print(x)
            if step % int(1/stepper.h) == 0:
                xs.append(x[None,:])
        return torch.cat(xs)
    
#%% Declare the model functions

odemlpfunc = ODEMLPFunc()
stepper = SingleStep(odemlpfunc, h)
model = MultiStep(stepper)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Define custom loss function to handle complex valued inputs
def loss_fcn(output, target, coeffs, lam=1e-3):
    # Custom loss function to handle complex valued inputs
    # Since we are computing a real valued pde, 
    # compute the inverse fft and compare 
    
    err = ( output - target )
    # Sum of squared modulus of coefficients in k-space is equivalent
    # to real space sum of squares by Plancherel theorem
    return torch.mean(err.real **2 + err.imag**2) + lam * (torch.abs(coeffs)).sum()

def loss_fcn(output, target):
    # Custom loss function to handle complex valued inputs
    # Since we are computing a real valued pde, 
    # compute the inverse fft and compare 
    
    err = ( output - target )
    # Sum of squared modulus of coefficients in k-space is equivalent
    # to real space sum of squares by Plancherel theorem
    return torch.mean(err.real **2 + err.imag**2)

#%%
# outputs = model(inputs, int((train_data.size(0)-1) / h))

#%% Train the model

# Training loop
epochs = 500
patience = 10
counter = 0
best_loss = np.inf
best_dict = model.state_dict()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs, int((train_data.size(0)-1) / h))
    assert outputs.size() == targets.size() , f'Output size {outputs.size()} does not match target size {targets.size()}'

    # coeffs = model.stepper.odefunc.return_coeffs()
    # print(coeffs)

    # loss = loss_fcn(outputs, targets, coeffs)
    loss = loss_fcn(outputs, targets)
    print(f'Epoch {epoch}, Loss {loss.item()}')
    with torch.no_grad():
        # val_loss = loss_fcn(model(val_i, int((val_data.size(0)-1) / h)), val_o, coeffs)
        val_loss = loss_fcn(model(val_i, int((val_data.size(0)-1) / h)), val_o)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    if val_loss < best_loss:
        best_dict = model.state_dict()
        best_loss = val_loss
        counter = 0

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
        print(f'Val Loss {val_loss.item()}')
    
    if counter > patience:
        "ran out of patience"
        break

#%% Validate the model

model.load_state_dict(best_dict)
with torch.no_grad():
    predicted_data = model(inputs, nmax-1).numpy().flatten()
    # predicted_data = np.insert(predicted_data, 0, 1, axis=0)
    # real_data = generate_data([1.0], a_true, dt, test_steps)

plt.close('all')
plt.plot(predicted_data, label='Predicted Solution')
plt.plot(real_data.flatten(), label='True Data')
plt.xlabel('Time step')
plt.ylabel('Solution (x)')
plt.title('Comparison of True Data and Predicted Solution')
plt.legend()
plt.grid(True)
plt.show()

# %%
