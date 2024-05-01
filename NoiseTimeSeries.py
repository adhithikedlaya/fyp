import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math
# Determine the number of points, n, and the length, T
def generate_pink_noise(n, T, alpha, beta, scaling_factor=0.005):
    """"This also determines the spectral space with wave numbers -k_max to k_max which correspond to frequency values f_k=kT/(2𝜋)
        .
        
        . 
        , ensure symmetry if you want real valued noise.
        for k=0..k_max

        
        ,
        C_k=𝐶𝑘𝑒𝑖𝜑𝑘
        ,
        C_-k=𝐶−𝑘𝑒−𝑖𝜑𝑘
        Take an inverse Fourier transform of the spectral coeficients to get your noise sequence, {𝑦𝑖}=ifft({𝐶𝑘})"""
    
    #This also determines the spectral space with wave numbers -k_max to k_max
    k_max = n // 2
    # which correspond to frequency values f_k=kT/(2𝜋)
    f_k = torch.linspace(0, k_max, steps=k_max+1) * T / (2 * np.pi)
    f_k[0] = 1e-10  # Avoid division by zero

    #Set the magnitudes of your spectral coefficients: C_k=1/|f_k|
    C_k = alpha *  f_k ** (-1 * beta)#plot csd of data against this in semi y log plot -> should be alpha beta, check that real and sampled match for different values

    #Set C_0=0 to give zero mean to the noise sequence.
    C_k[0] = 0  # Zero mean

    #Set the phases of the spectral coefficients to random values - 𝜙𝑘=( rand∈[0,2𝜋) )
    phi_k = torch.rand(k_max+1) * 2 * np.pi

    #C_k=𝐶𝑘𝑒𝑖𝜑𝑘
    C_k_real = C_k * torch.cos(phi_k)
    C_k_imag = C_k * torch.sin(phi_k)

    # Halve to preserve mean of 0 and 1/f spectrum?
    C_k_real[1:k_max+1] *= 0.5
    C_k_imag[1:k_max+1] *= 0.5
    C_k_complex_pos = torch.complex(C_k_real, C_k_imag)
    
    #C_-k=𝐶−𝑘𝑒−𝑖𝜑𝑘
    # Reverse and conjugate for negative frequencies - symmetrical = real valued noise
    C_k_complex_neg = torch.flip(C_k_complex_pos[1:k_max], dims=[0]).conj()

    # Concat negative and positive values
    C_k_complex = torch.cat((C_k_complex_pos, C_k_complex_neg))

    # Take an inverse Fourier transform of the spectral coeficients to get your noise sequence
    noise_time_domain = fft.irfft(C_k_complex, n)
    
    # Normalize noise to have unit variance - optional addition not in orignal algo?
    noise_time_domain /= torch.std(noise_time_domain)

    return noise_time_domain * scaling_factor, C_k_complex

def plot_noise_time_domain(noise_sequence, T):
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, T, len(noise_sequence)), noise_sequence)
    plt.title("Pink Noise in Time Domain")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_noise_freq_domain(noise_freq_domain, T):
    plt.figure(figsize=(10, 5))
    freqs = torch.fft.fftfreq(len(noise_freq_domain), T / len(noise_freq_domain))
    plt.plot(freqs[:len(noise_freq_domain) // 2], torch.abs(noise_freq_domain)[:len(noise_freq_domain) // 2])
    plt.title("Pink Noise in Frequency Domain")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

def plot_noise_freq_domain_log(noise_freq_domain, T):
    plt.figure(figsize=(10, 5))
    freqs = torch.fft.fftfreq(len(noise_freq_domain), T / len(noise_freq_domain))
    plt.semilogy(freqs[:len(noise_freq_domain) // 2], torch.abs(noise_freq_domain)[:len(noise_freq_domain) // 2])
    # freqs = torch.fft.fftfreq(len(noise_freq_domain), T / len(noise_freq_domain))
    # freqs = freqs[:len(noise_freq_domain) // 2]
    # noise_freq_domain = torch.abs(noise_freq_domain)[:len(noise_freq_domain) // 2]
    # plt.plot(torch.log(freqs), torch.log(noise_freq_domain) )
    plt.title("Pink Noise in Frequency Domain")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

# Example usage
# n_points = 500
# sequence_length = 1.0  # arbitrary length for demonstration
# noise_sequence, noise_freq_domain = generate_pink_noise(n_points, sequence_length, 1, 1)

# plot_noise_time_domain(noise_sequence, sequence_length)
# plot_noise_freq_domain(noise_freq_domain, sequence_length)
