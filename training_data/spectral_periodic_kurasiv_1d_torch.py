import torch as th
from torch.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Spatial grid and initial condition
N = 128
x = 32*th.pi*th.arange(1,N+1)/N
u = th.cos(x/16)*(1+th.sin(x/16))
v = fft(u)

# Precompute ETDRK4 scalar quantities
h = 1/4
k = th.cat([th.arange(0,N/2),th.tensor([0.]),th.arange(-N/2+1,0)],0)/16
L = k**2 - k**4 
E = (h*L).exp()
E2 = (h*L/2).exp()
M = 16
r = (1j*th.pi*(th.arange(1,M+1)-.5)/M)
LR = h*L[:,None].repeat_interleave(M,1) + r[None,:].repeat_interleave(N,0)
Q = h*(((LR/2).exp()-1)/LR).mean(dim=1).real
f1 = h*((-4-LR+LR.exp()*(4-3*LR+LR**2))/LR**3).mean(dim=1).real
f2 = h*((2+LR+LR.exp()*(-2+LR))/LR**3).mean(dim=1).real
f3 = h*((-4-3*LR-LR**2+LR.exp()*(4-LR))/LR**3).mean(dim=1).real

# Timestepping
uu = [u]
vv = [v]
tt = [0]
tmax = 1000
nmax = int(tmax/h)
nplt = int((tmax/tmax)/h)
g = -.5j*k

for n in range(1,nmax+1):
    t = n*h
    Nv = g * fft(ifft(v).real**2)
    a = E2*v + Q*Nv
    Na = g * fft(ifft(a).real**2)
    b = E2*v + Q*Na
    Nb = g * fft(ifft(b).real**2)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = g * fft(ifft(c).real**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    if n%nplt==0:
        vv.append(v)
        u = ifft(v).real
        uu.append(u)
        tt.append(t) 

vv = th.stack(vv)
# Save the fourier space solution 
fn = f'ks_soln_ft_N_{N}_dt_{h}_tmax_{tmax}'
th.save(vv, fn + '.pt')
readme = f"Stored solution data for 1d-KS equation with N = {N}, dt = {h}, tmax = {tmax}.\
        \nTo load the solution, use the following command:\
        \n\nvv = th.load('ks_soln_ft_N_{N}_dt_{h}_tmax_{tmax}.pt')\
        \n\n Data were stored every int(1/h) steps."

with open(fn + '.txt', 'w') as f:
    f.write(readme)

uu = th.stack(uu)
tt = th.tensor(tt)

# Create a figure and axis for the animation
plt.close('all')
fig, ax = plt.subplots(num=1, clear=True)
ax.plot(x, uu[0], lw=2)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-10,10)
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
# ax.set_title()
ax.grid(True)
# plt.show()
# exit()
# Animation update function
def update(frame):
    ax.clear()
    ax.plot(x, uu[frame], lw=2)
    # ax.plot(x, np.fft.fftshift(np.fft.ifft(soln_store_linear[frame])).real
    #             , lw=2, label='Linear')

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-10,10)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    # ax.set_title('Solution of 1D Heat Equation Animation')
    ax.grid(True)
    # ax.legend(fontsize=9)
    return ax,

# Create and display the animation
ani = FuncAnimation(fig, update, frames=len(uu), blit=False, interval=300)
ani.save('spectral_periodic_kurasiv_1d_torch.gif', writer='pillow')
plt.show()