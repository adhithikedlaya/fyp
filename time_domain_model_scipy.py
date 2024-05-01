import numpy as np
from scipy.optimize import minimize
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward import getEulerBOLD
from scipy import signal
import math

def plot_csds(f, csdx, csdy):
    plt.xlim(0, 1)
    plt.plot(f, csdx, label="predicted")
    plt.plot(f, csdy, label="observed")
    plt.legend()
    plt.show()

def custom_loss(parameters, observed_signal):
    # Generate signal using the function with the given parameters
    predicted_signal = forward(parameters)
    
    # Calculate the mean squared error as the loss using CSD???
    f, csdx = signal.csd(predicted_signal.detach().numpy(), predicted_signal.detach().numpy(), fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    f, csdy = signal.csd(observed_signal.detach().numpy(), observed_signal.detach().numpy(), fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    
    # plot_csds(f, csdx, csdy)
    print(csdx.shape)
    # Compute the mean squared error between the cross-spectral densities
    mse_real = np.mean(np.abs(np.real(csdx) - np.real(csdy))**2)
    mse_imaginary = np.mean(np.abs(np.imag(csdx) - np.imag(csdy))**2)
    
    # Combine the losses for real and imaginary parts
    loss = mse_real + mse_imaginary
    print(loss * 100000)
    return loss * 100000
    
def forward(parameters):
    # Call the EULER BOLD function using chosen alpha and beta noise parameters (TODO: add MTT as a param too)
    _, yhat = getEulerBOLD(parameters[0], parameters[1], parameters[2], noise=True, length=1000)
    return yhat

# Use simulated signal as ground truth
_, observed_signal = getEulerBOLD(2, 1, 1, True)

# Initial parameters
initial_parameters = [1, 1, 1]

# Minimize the loss function to learn the parameters
result = minimize(custom_loss, initial_parameters, args=(observed_signal,))
learned_parameters = result.x

print("Learned parameters:", learned_parameters)