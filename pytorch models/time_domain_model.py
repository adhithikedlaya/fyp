import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward import getEulerBOLD
from signal_pytorch import csd
import json
import sys
import numpy as np
from scipy.signal import csd as scipy_csd
from torch.optim.lr_scheduler import ExponentialLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def csd_no_window(x, y, nfft=4096):
        # Compute Fourier transforms of the signals
    X = torch.fft.fft(x, n=nfft)
    Y = torch.fft.fft(y, n=nfft)
    
    # Compute cross-spectral density
    cross_spectrum = X * torch.conj(Y)
    
    return cross_spectrum
  
x_train_tensor, y_train_tensor = getEulerBOLD(torch.tensor(2.0, requires_grad=True), torch.tensor(0.8, requires_grad=True), torch.tensor(0.3, requires_grad=True), torch.tensor(0.5, requires_grad=True), torch.tensor(0.6, requires_grad=True), 1.0, torch.tensor(1.0, requires_grad=True), True, 1000)
# y_train_tensor = np.loadtxt('./time_series/sub-001-PLCB-ROI0.txt', delimiter=',')
f2, csdy = csd(y_train_tensor, y_train_tensor, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
# f2, csdy = csd(target, target, fs=0.5, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)

class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # assuming we are fitting noise parameters for now? - how would multiple regions work?
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))
        self.mtt = nn.Parameter(torch.tensor(1.5))
        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.4))
        self.lamb = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(0.25))
        self.beta = nn.Parameter(torch.tensor(1.0))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(self.mtt, self.sigma, self.mu, self.lamb, self.c, 1.0, self.beta, True, 1000)
        yhat = torch.stack(yhat)
        return yhat
        # return torch.tensor(yhat, requires_grad=True)

    
model = TimeDomainModel().to(device)

lr = 0.2
n_epochs = 20


def complex_mse_loss(output):
    # downsampled_output = output[::200][:len(target)]
    # downsampled_output = output
    # plt.plot(np.arange(len(output)), [x.detach().numpy() for x in downsampled_output], label='simulated BOLD')
    # plt.plot(np.arange(len(target)), [x.detach().numpy() for x in target], label='real BOLD')
    # plt.legend()
    # plt.show()
    f1, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    # f3, csd_d = csd(downsampled_output, downsampled_output, fs=0.5, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)
    # plt.xlim(0, 3)
    # plt.plot(f1, [x.detach().numpy() for x in csdx], label='simulated')
    # plt.legend()
    # plt.show()
    # plt.xlim(0, 5)
    # plt.plot(f1, csdy, label='true value')
    # plt.legend()
    # plt.show()
    # plt.xlim(0, 1)
    # plt.plot(f1, csd_d, label='simulated downsampled')
    # plt.legend()
    # plt.show()
    # Compute the mean squared error between the cross-spectral densities
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)

    # Combine the losses for real and imaginary parts
    loss = mse_real + mse_imaginary
    return loss


def train():
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.train()
    losses = []
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        yhat = model()
        loss = complex_mse_loss(yhat)
        print("Loss: ", loss)
        print("Params: ", model.mtt, model.sigma, model.mu, model.lamb, model.c, model.beta)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        

    # Save losses array as JSON
    with open('losses.json', 'w') as f:
        json.dump(losses, f)

    # Save final model's parameters (MTT) in a text file
    with open('final_mtt.txt', 'w') as f:
            f.write(str(model.mtt.item()))

    print(model.mtt)

torch.set_printoptions(precision=8)
train()

#main needs to take in a subject number, lsd or plcb and roi
# if __name__ == "__main__":
#      subj = f'sub-0{sys.argv[1]}'
#      exp = f'ses-{sys.argv[2]}'
#      roi = int(sys.argv[3])



# all_losses = []
# betas = torch.arange(0.5, 3.5, 0.1)
# mtts = torch.arange(3, 10, 0.5)
# num_sims = 10

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