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

num_regions = 1
def h(omega):
    i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    hrf = (6 * (i * omega + 1) ** 10 - 1) / (5 * (i * omega + 1) ** 16)
    return hrf * torch.eye(num_regions)

def g(omega, alphas, betas):
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
    csd_curves = torch.empty((num_regions ** 2, freqs.size(0)), dtype=torch.complex64)
    for (i, freq) in enumerate(freqs):
        csds = forward_single_freq(freq).view(-1)
        for (reg, csd_val) in enumerate(csds):
            csd_curves[reg, i] = csd_val  
    return csd_curves

    
def plot(gs):
    plt.figure(figsize=(10, 5))    
    y = y_train_tensor.detach().numpy()
    g = gs[0].detach().numpy()
    print(g)
    plt.plot(f, g, label="simulated CSD")
    plt.plot(f, y, label="empirical CSD")
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()
    return
    

def complex_mse_loss(output, target):
    loss =  torch.abs(((output.real - target.real)**2).sum())
    return loss 

alpha_min = -2
alpha_max = 2
beta_min = 0.5
beta_max = 3.5
a_min = -2
a_max = -0.1
step = 0.1



a = a_min
alpha = alpha_min
beta = beta_min
best_loss = 934829304829348
best_beta = beta
best_alpha = alpha
best_a = a
for a in np.arange(a_min, a_max, step):
    A = torch.tensor([a], dtype=torch.float)
    for alpha in np.arange(alpha_min, alpha_max, step):
        if alpha != 0:
            alphas_e = torch.tensor([alpha])
            alphas_v = alphas_e
            for beta in np.arange(beta_min, beta_max, step):
                betas_e = torch.tensor([beta])
                betas_v = betas_e
                yhat = forward(x_train_tensor)
                loss = complex_mse_loss(yhat, y_train_tensor)
                if loss<=best_loss:
                    best_loss = loss
                    best_beta = beta
                    best_alpha = alpha
                    best_a = a
                print(loss, alpha, beta, a)

A = torch.tensor([best_a])
betas_e = torch.diag(torch.tensor([best_beta]))
betas_v = betas_e
alphas_e = torch.diag(torch.tensor([best_alpha]))
alphas_v = alphas_e
yhat = forward(x_train_tensor)
print("best params" ,best_a, best_alpha, best_beta, best_loss)
plot(yhat)