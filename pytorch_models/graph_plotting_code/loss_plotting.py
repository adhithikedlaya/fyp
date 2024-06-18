import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward import getEulerBOLD
from signal_pytorch import csd
import json
import sys
import numpy as np
from scipy.signal import csd as scipy_csd
from torch.optim.lr_scheduler import ExponentialLR

def complex_mse_loss(csdy, output):
    f1, csdx = csd(output, output, fs=100,nperseg=20000)
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real + mse_imaginary
    return loss


def plot_heatmap(vars1, vars2):

    var1, vals1 = vars1
    var2, vals2 = vars2

    x_train_tensor, observed_bold  = getEulerBOLD(phi=torch.tensor(1.9), noise=True, length=1000)

    _, csdy = csd(observed_bold, observed_bold, fs=100, nperseg=20000)

    variables_dict = {
        'beta': 1.0,
        'mu': 0.4,
        'sigma': 0.5,
        'lamb': 0.2, 
        'phi': 1.9,
        'psi': 0.6, 
        'chi': 0.6}
    num_sims = 1

    all_losses = np.zeros((len(vals1), len(vals2)))
    for i in range(num_sims):
        print(i)
        losses = []
        for j, val1 in enumerate(vals1):
            for k, val2 in enumerate(vals2):
                variables_dict[var1] = val1
                variables_dict[var2] = val2
                _, bold = getEulerBOLD(
                            sigma=torch.tensor(variables_dict['sigma']),
                            mu=torch.tensor(variables_dict['mu']),
                            lamb=torch.tensor(variables_dict['lamb']),
                            alpha=1.0,
                            beta=torch.tensor(variables_dict['beta']),
                            phi=torch.tensor(variables_dict['phi']),
                            psi=torch.tensor(variables_dict['psi']),
                            chi=torch.tensor(variables_dict['chi']),
                            noise=True,
                            length=1000
                        )
                bold = torch.stack(bold)
                loss = complex_mse_loss(csdy, bold)
                print(loss)
                all_losses[j, k] += loss.detach().numpy()


    all_losses /= num_sims
    print(all_losses)
    np.savetxt(f"loss_graph_{var1}_{var2}.txt", all_losses, delimiter=',')

    # Plot heatmap
   # plt.imshow(all_losses, extent=[vals1[0], vals1[-1], vals2[0], vals2[-1]], aspect='auto', origin='lower')
   # plt.colorbar(label='Mean loss')
   # plt.xlabel(var1)
   # plt.ylabel(var2)
   # plt.title('Mean loss heatmap over 25 simulations')
   # plt.show()

betas = np.arange(0.5, 1.5, 0.1)
mus = np.arange(0.7, 1.3, 0.1)
lambdas = np.arange(0.2, 0.6, 0.1)
sigmas = np.arange(0.2, 0.9, 0.1)
phis = np.arange(1.6, 2.4, 0.1)
psis = np.arange(0.3, 1.0, 0.1)
#plot_heatmap(('lamb', lambdas), ('sigma', sigmas))
# plot_heatmap(('lamb', lambdas), ('mu', mus))
plot_heatmap(('phi', phis), ('psi', psis))
