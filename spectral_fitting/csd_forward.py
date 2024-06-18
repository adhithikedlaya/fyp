import torch
import torch.optim as optim
import torch.nn as nn
from csd_calculation import csds, f
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor = torch.from_numpy(f).float().to(device) 
y_train_tensor = torch.complex(torch.from_numpy(csds[0].real), torch.from_numpy(csds[0].imag)).to(device) 

betas_e = torch.tensor([-0.5])
alphas_e = torch.tensor([0.4])
betas_v = torch.tensor([-0.4])
alphas_v = torch.tensor([-2.6])
A = torch.tensor([-3])

num_regions = 1
def h(omega):
    i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    hrf = (6 * (i * omega + 1) ** 10 - 1) / (5 * (i * omega + 1) ** 16)
    return hrf * torch.eye(num_regions)

def g(omega, alphas, betas):
    # return (1/((omega + 1) ** 2)) * torch.eye(num_regions)
    return torch.diag(alphas * torch.full([num_regions], omega) ** (-1 * betas))


def forward_single_freq(omega):
    I = torch.eye(num_regions)
    i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    hrf = h(omega)
    G_v = g(omega, alphas_v, betas_v) 
    G_v = G_v.to(torch.complex64)
    X = i * omega * I - A
    X = X.to(torch.complex64)
    C = torch.linalg.solve(X, hrf, left=False)
    G_e = g(omega, alphas_e, betas_e) 
    result =  C @ G_v @ torch.conj(C).T + G_e
    return result

def forward(freqs):
    csd_curves = torch.empty((num_regions ** 2, freqs.size), dtype=torch.complex64)
    for (i, freq) in enumerate(freqs):
        csds = forward_single_freq(freq).view(-1)
        for (reg, csd_val) in enumerate(csds):
            csd_curves[reg, i] = csd_val  
    return csd_curves


simulated_csds = forward(f)