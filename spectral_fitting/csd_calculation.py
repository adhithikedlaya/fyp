
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


ws = [2, 50, 75, 100, 175]
nperseg = np.floor(len(pcc_data)/8)

csds = []
for i in range(4):
    for j in range(4):
        region1 = data[:, i]
        region2 = data[:, j]
        f, csd = signal.csd(region1, region2, fs=0.5, nperseg=nperseg, noverlap=None, nfft=4096, window='hamming', scaling='density')
        csds.append(csd[1:])


f = f[1:]
csds = np.array(csds)
