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
    predicted_signal = forward(parameters)
    
    f, csdx = signal.csd(predicted_signal, predicted_signal, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    f, csdy = signal.csd(observed_signal, observed_signal, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    

    mse_real = np.mean(np.abs(np.real(csdx) - np.real(csdy))**2)
    mse_imaginary = np.mean(np.abs(np.imag(csdx) - np.imag(csdy))**2)
    

    loss = mse_real + mse_imaginary
    return loss
    
def forward(parameters):
    _, yhat = getEulerBOLD(parameters[0], noise=True, length=1000)
    return yhat

# Use simulated signal as ground truth
_, observed_signal = getEulerBOLD(2, 1, 1, True, length=1000)


param_space = [Real(low=0.0, high=5.0, name='mtt'),
               # Add more parameters as needed
              ]

start_time = time.perf_counter()

result = gp_minimize(custom_loss,
                     param_space,
                     acq_func="gp_hedge",
                     n_calls=40,
                     n_random_starts=3,
                     random_state=1234)

optimized_params = result.x
end_time = time.perf_counter()
print(end_time - start_time)

print("Learned parameters:", optimized_params)