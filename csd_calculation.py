
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data = loadmat("data.mat")['data']
csd = loadmat("csd.mat")['data']
pcc_data = data[:, 0]
mpfc_data = data[:, 1]

nperseg = 1000
noverlap = 1  # You can experiment with different values
nfft = 10000 # You can experiment with different values


window_sizes = [16, 32, 64, 128]
overlaps = [2, 4, 8, 16, 32, 64, 128]

# plt.figure(figsize=(12, 8))

# plt.xlim(0, 0.15)
# plt.ylim(0, 1.31)
ws = [2, 50, 75, 100, 175]
nperseg = np.floor(len(pcc_data)/8)
f, csd1 = signal.csd(pcc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
f, csd2 = signal.csd(pcc_data, mpfc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
f, csd3 = signal.csd(mpfc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
f, csd4 = signal.csd(mpfc_data, mpfc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
f = f[1:1000]
csd = np.array([csd1[1:1000], csd2[1:1000], csd3[1:1000], csd4[1:1000]])

# plt.plot(f, csd.real)
# plt.plot(f, csd.imag)
# plt.legend()
# plt.title('fMRI data - Cross Spectral Density')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
#plt.show()

# attempt at calculating fourier transform and then calculating CSD from that but i get a dimension error
# xs = np.arange(175)
# freq = xs / 175
# ft = fft(data)
# csd = np.matmul(ft, ft.conj().T)
# print(csd)

# for nperseg in window_sizes:
#     for noverlap in overlaps:
#         if noverlap < nperseg:
#             print(noverlap, nperseg)
#             f, csd = signal.csd(pcc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=noverlap, nfft=nperseg*200, window='hamming', detrend=False)
#             plt.plot(f, csd.real, label=str(nperseg) + ", " + str(noverlap))

# np.show()
