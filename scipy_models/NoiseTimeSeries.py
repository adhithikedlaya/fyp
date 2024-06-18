import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math

def generate_pink_noise(n, T, alpha, beta, scaling_factor=0.005):
    k_max = n // 2
    f_k = torch.linspace(0, k_max, steps=k_max+1) * T / (2 * np.pi)
    f_k[0] = 1e-10  # Avoid division by zero

    #Set the magnitudes of spectral coefficients
    C_k = alpha *  f_k ** (-1 * beta)
    C_k[0] = 0

    #Set the phases of the spectral coefficients to random values
    phi_k = torch.rand(k_max+1) * 2 * np.pi
    C_k_real = C_k * torch.cos(phi_k)
    C_k_imag = C_k * torch.sin(phi_k)
    C_k_real[1:k_max+1] *= 0.5
    C_k_imag[1:k_max+1] *= 0.5
    C_k_complex_pos = torch.complex(C_k_real, C_k_imag)
    

    # Reverse and conjugate for negative frequencies
    C_k_complex_neg = torch.flip(C_k_complex_pos[1:k_max], dims=[0]).conj()
    C_k_complex = torch.cat((C_k_complex_pos, C_k_complex_neg))

    # Take an inverse Fourier transform of the spectral coeficients to get noise sequence
    noise_time_domain = fft.irfft(C_k_complex, n)
    noise_time_domain = noise_time_domain / torch.std(noise_time_domain)

    return noise_time_domain * scaling_factor, C_k_complex