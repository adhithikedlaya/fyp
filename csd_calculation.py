
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data = loadmat("data.mat")['data']
csd = loadmat("csd.mat")['data']
pcc_data = data[:, 0]
mpfc_data = data[:, 1]
lipc_data = data[:, 2]
ripc_data = data[:, 3]

nperseg = 1000
noverlap = 1  # You can experiment with different values
nfft = 10000 # You can experiment with different values


window_sizes = [16, 32, 64, 128]
overlaps = [2, 4, 8, 16, 32, 64, 128]

# plt.figure(figsize=(12, 8))

plt.xlim(0, 0.15)
plt.ylim(0, 1.31)
ws = [2, 50, 75, 100, 175]
nperseg = np.floor(len(pcc_data)/8)

csds = []
for i in range(4):
    for j in range(4):
        region1 = data[:, i]
        region2 = data[:, j]
        f, csd = signal.csd(region1, region2, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
        csds.append(csd[1:])
# f, csd1 = signal.csd(pcc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd2 = signal.csd(mpfc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd3 = signal.csd(lipc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd4 = signal.csd(ripc_data, pcc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd5 = signal.csd(mpfc_data, mpfc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd6 = signal.csd(lipc_data, mpfc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd7 = signal.csd(ripc_data, mpfc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd8 = signal.csd(lipc_data, lipc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd9 = signal.csd(ripc_data, lipc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
# f, csd10 = signal.csd(ripc_data, ripc_data, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')


f = f[1:]
csds = np.array(csds)

# plt.plot(f, csd1.real, label='real')
# plt.plot(f, csd1.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - PCC to PCC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd2.real, label='real')
# plt.plot(f, csd2.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - mPFC to PCC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd3.real, label='real')
# plt.plot(f, csd3.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - LIPC to PCC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd4.real, label='real')
# plt.plot(f, csd4.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - RIPC to PCC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd5.real, label='real')
# plt.plot(f, csd5.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - mPFC to mPFC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd6.real, label='real')
# plt.plot(f, csd6.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - LIPC to mPFC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd7.real, label='real')
# plt.plot(f, csd7.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - RIPC to mPFC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd8.real, label='real')
# plt.plot(f, csd8.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - LIPC to LIPC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd9.real, label='real')
# plt.plot(f, csd9.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - RIPC to LIPC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

# plt.plot(f, csd10.real, label='real')
# plt.plot(f, csd10.imag, label='imaginary')
# plt.legend()
# plt.title('Empirical Cross Spectral Density - RIPC to RIPC')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()

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
