import numpy as np
from scipy.optimize import minimize
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward_Scipy import getEulerBOLD
from scipy import signal
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def plot_csds(f, csdx, csdy):
    plt.xlim(0, 1)
    plt.plot(f, csdx, label="predicted")
    plt.plot(f, csdy, label="observed")
    plt.legend()
    plt.show()

def custom_loss(parameters):
    # Generate signal using the function with the given parameters
    predicted_signal = forward(parameters)
    
    # Calculate the mean squared error as the loss using CSD???
    f, csdx = signal.csd(predicted_signal, predicted_signal, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    f, csdy = signal.csd(observed_signal, observed_signal, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    
    # plot_csds(f, csdx, csdy)
    print(csdx.shape)
    # Compute the mean squared error between the cross-spectral densities
    mse_real = np.mean(np.abs(np.real(csdx) - np.real(csdy))**2)
    mse_imaginary = np.mean(np.abs(np.imag(csdx) - np.imag(csdy))**2)
    
    # Combine the losses for real and imaginary parts
    loss = mse_real + mse_imaginary
    print("Loss: ", loss * 100000)
    print("Parameters: ", parameters)
    return loss
    
def forward(parameters):
    # Call the EULER BOLD function using chosen alpha and beta noise parameters (TODO: add MTT as a param too)
    _, yhat = getEulerBOLD(parameters[0], noise=True, length=1000)
    return yhat

# Use simulated signal as ground truth
_, observed_signal = getEulerBOLD(2, 1, 1, True, length=1000)


# Minimize the loss function to learn the parameters

param_space = [Real(low=0.0, high=5.0, name='mtt'),
               # Add more parameters as needed
              ]

start_time = time.perf_counter()
# Run optimization
result = gp_minimize(custom_loss,                  # the function to minimize
                     param_space,                  # the bounds on each dimension of x
                     acq_func="gp_hedge",               # the acquisition function
                     n_calls=40,                  # the number of evaluations of f
                     n_random_starts=3,           # the number of random initialization points
                     random_state=1234)            # the random seed

# Result will contain the optimized parameters and the minimum value found
optimized_params = result.x
end_time = time.perf_counter()
print(end_time - start_time)

# start_time = time.perf_counter()
# result = minimize(custom_loss, initial_parameters, args=(observed_signal,), method='Powell', tol=1e-4)
# end_time = time.perf_counter()
# learned_parameters = result.x
# print(end_time - start_time)

print("Learned parameters:", optimized_params)