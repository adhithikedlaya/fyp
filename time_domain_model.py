import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward import getEulerBOLD
import numpy as np
from scipy import signal
from signal_pytorch import csd
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def csd_no_window(x, y, nfft=4096):
        # Compute Fourier transforms of the signals
    X = torch.fft.fft(x, n=nfft)
    Y = torch.fft.fft(y, n=nfft)
    
    # Compute cross-spectral density
    cross_spectrum = X * torch.conj(Y)
    
    return cross_spectrum
  
x_train_tensor, y_train_tensor = getEulerBOLD(torch.tensor(2.0, requires_grad=True), 1.0, 1.0, True, 1000)

class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # assuming we are fitting noise parameters for now? - how would multiple regions work?
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))
        self.mtt = nn.Parameter(torch.tensor(1.0))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(self.mtt, 1.0, 1.0, True, 1000)
        yhat = torch.stack(yhat)
        return yhat
        # return torch.tensor(yhat, requires_grad=True)

    
model = TimeDomainModel().to(device)

lr = 0.01
n_epochs = 150

def complex_mse_loss(output, target):
    _, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    _, csdy = csd(target, target, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096)
    # Compute the mean squared error between the cross-spectral densities
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)

    # Combine the losses for real and imaginary parts
    loss = mse_real + mse_imaginary
    return loss

optimizer = optim.Adam(model.parameters(), lr=lr)
model.train()
losses = []
for epoch in range(n_epochs):
    print("Epoch: ", epoch)
    yhat = model()
    loss = complex_mse_loss(yhat, y_train_tensor)
    print(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    losses.append(loss.item())
    

# Save losses array as JSON
with open('losses.json', 'w') as f:
    json.dump(losses, f)

# Save final model's parameters (MTT) in a text file
with open('final_mtt.txt', 'w') as f:
        f.write(model.mtt.item())

print(model.mtt)

# plt.xlabel('MTT value')
# plt.ylabel('Loss')
# plt.title('Mean loss curve over 100 simulations')
# plt.plot(mtts, mean_loss, label='Mean loss')
# plt.legend()
# plt.fill_between(mtts, mean_loss-std_deviation_loss, mean_loss+std_deviation_loss, facecolor = 'lightblue', label='standard deviation')
# plt.show()


# all_losses = []
# betas = torch.arange(0.5, 3.5, 0.1)
# mtts = torch.arange(1, 3, 0.1)
# num_sims = 50

# for i in range(num_sims):
#     print(i)
#     losses = []
#     for mtt in mtts:
#         _, bold = getEulerBOLD(mtt, 1, 1, True, 1000)
#         # plot_csds(f, csdx, csdy)
#         loss = complex_mse_loss(bold, y_train_tensor)
#         losses.append(loss)
#     all_losses.append(losses)

# all_losses = np.array(all_losses)

# mean_loss = np.nanmean(all_losses, axis=0)
# std_deviation_loss = np.nanstd(all_losses, axis=0)

# plt.xlabel('MTT value')
# plt.ylabel('Loss')
# plt.title('Mean loss curve over 100 simulations')
# plt.plot(mtts, mean_loss, label='Mean loss')
# plt.legend()
# plt.fill_between(mtts, mean_loss-std_deviation_loss, mean_loss+std_deviation_loss, facecolor = 'lightblue', label='standard deviation')
# plt.show()