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
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor, y_train_tensor = getEulerBOLD(torch.tensor(2.0, requires_grad=True), 1.0, 1.0, True, 1000)

class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mtt = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        _, yhat = getEulerBOLD(self.mtt, 1.0, 1.0, True, 1000)
        yhat = torch.stack(yhat)
        return yhat

    
model = TimeDomainModel().to(device)

lr = 0.01
n_epochs = 100


def complex_mse_loss(output, target):

    f1, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    f2, csdy = csd(target, target, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real + mse_imaginary
    return loss


def train():
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.train()
    losses = []
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        yhat = model()
        loss = complex_mse_loss(yhat, y_train_tensor)
        print("Loss: ", loss)
        print("Params: ", model.mtt)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
    with open('losses.json', 'w') as f:
        json.dump(losses, f)

    with open('final_mtt.txt', 'w') as f:
            f.write(str(model.mtt.item()))

    print(model.mtt)

torch.set_printoptions(precision=8)
start_time = time.perf_counter()
train()
end_time = time.perf_counter()
print(end_time - start_time)