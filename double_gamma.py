import scipy.stats
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from csd_calculation import csd, f
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from csd_calculation import f
x = np.arange(0, 50, 0.01)

def hrf(times):
     """ Return values for HRF at given times """
     # Gamma pdf for the peak
     peak_values = gamma.pdf(times, 6)
     # Gamma pdf for the undershoot
     undershoot_values = gamma.pdf(times, 15)
     # Combine them
     values = peak_values - 0.35 * undershoot_values
 # Scale max to 0.6
     hrf = values / np.max(values) * 5
     return fft(hrf)


def transfer(times):
    i = 1j
    return (6 * (i * times + 1) ** 10 - 1) / (5 * (i * times + 1) ** 16)

# plt.ylim(0, 1.31)
fft_freq = fftfreq(len(x), 0.01)
print(len(x))
h = hrf(x)
plt.plot(fft_freq, np.abs(h/(len(x))))
plt.xlim(0, 0.5)
#plt.plot(fft_freq, np.abs(transfer(fft_freq)), label="transfer")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Double-gamma mixture transfer function')
plt.show()