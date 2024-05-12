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
  
x_train_tensor, y_train_tensor = getEulerBOLD(torch.tensor(2.0, requires_grad=True), torch.tensor(0.8, requires_grad=True), torch.tensor(0.3, requires_grad=True), torch.tensor(0.5, requires_grad=True), torch.tensor(0.6, requires_grad=True), 1.0, torch.tensor(1.0, requires_grad=True), True, 1000)
f2, csdy = csd(y_train_tensor, y_train_tensor, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
# f2, csdy = csd(target, target, fs=0.5, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)
# y_train_tensor = np.loadtxt('./time_series/sub-001-PLCB-ROI0.txt', delimiter=',')

def get_torch_rand(min, max):
    return torch.rand(()) * (max - min) + min


class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mtt = nn.Parameter(get_torch_rand(1.0, 3.0))
        self.sigma = nn.Parameter(get_torch_rand(0.5, 1.0))
        self.mu = nn.Parameter(get_torch_rand(0.1, 0.5))
        self.lamb = nn.Parameter(get_torch_rand(0.3, 0.8))
        self.c = nn.Parameter(get_torch_rand(0.2, 1))
        self.beta = nn.Parameter(get_torch_rand(0.8, 1.2))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(self.mtt, self.sigma, self.mu, self.lamb, self.c, 1.0, self.beta, True, 1000)
        yhat = torch.stack(yhat)
        return yhat
        # return torch.tensor(yhat, requires_grad=True)

    



def complex_mse_loss(output):
    f1, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real + mse_imaginary
    return loss


def train():
    model = TimeDomainModel().to(device)
    lr = 0.2
    n_epochs = 20
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

def train_multiple():
    lr = 0.2
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    num_initializations = 5
    models = [TimeDomainModel().to(device) for _ in range(num_initializations)]

# Train each instance separately using gradient descent
    num_epochs = 10
    model_losses = []
    for model in models:
        losses = []
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model()
            loss = complex_mse_loss(outputs)
            print(loss)
            losses.append(loss.item())
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model_losses.append(losses)

    # Evaluate the performance of each instance based on the final loss

    # Select the instance with the lowest loss
    best_model_idx = min(range(len(model_losses)), key=lambda i: model_losses[i][-1])
    print(best_model_idx, model_losses[best_model_idx][-1])

torch.set_printoptions(precision=8)
train_multiple()