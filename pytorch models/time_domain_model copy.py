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
  
x_train_tensor, y_train_tensor = getEulerBOLD(sigma=torch.tensor(0.8, requires_grad=True), mu=torch.tensor(0.3, requires_grad=True), lamb=torch.tensor(0.5, requires_grad=True), alpha=1.0, beta=torch.tensor(1.0, requires_grad=True), noise=True, length=1000)
f2, csdy = csd(y_train_tensor, y_train_tensor, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
# f2, csdy = csd(target, target, fs=0.5, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)
# y_train_tensor = np.loadtxt('./time_series/sub-001-PLCB-ROI0.txt', delimiter=',')

def get_torch_rand(min, max):
    return torch.rand(()) * (max - min) + min


class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(get_torch_rand(0.5, 1.0))
        self.mu = nn.Parameter(get_torch_rand(0.1, 0.5))
        self.lamb = nn.Parameter(get_torch_rand(0.3, 0.8))
        self.beta = nn.Parameter(get_torch_rand(0.8, 1.2))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(sigma=self.sigma, mu=self.mu, lamb=self.lamb, alpha=1.0, beta=self.beta, noise=True, length=1000)
        yhat = torch.stack(yhat)
        return yhat
        # return torch.tensor(yhat, requires_grad=True)

    



def complex_mse_loss(output, csdy):
    f1, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real + mse_imaginary
    return loss


def train(observed_csd):
    model = TimeDomainModel().to(device)
    lr = 0.01
    n_epochs =30
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model.train()
    losses = []
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        yhat = model()
        loss = complex_mse_loss(yhat, observed_csd)
        print("Loss: ", loss)
        print("Params: ", model.sigma, model.mu, model.lamb, model.beta)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        

    # Save losses array as JSON
    with open('losses.json', 'w') as f:
        json.dump(losses, f)

def train_multiple(observed_csd, name):
    lr = 0.01
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    num_initializations = 3
    models = [TimeDomainModel().to(device) for _ in range(num_initializations)]

# Train each instance separately using gradient descent
    num_epochs = 30
    model_losses = []
    for i, model in enumerate(models):
        losses = []
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            print(f"Model {i} Epoch: {epoch}")
            # Forward pass
            outputs = model()
            loss = complex_mse_loss(outputs, observed_csd)
            print("Loss: ", loss)
            print("Params: ", model.sigma, model.mu, model.lamb, model.beta)
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
    best_model = models[best_model_idx]
    print("Params: ", best_model.sigma, best_model.mu, best_model.lamb, best_model.beta)
    with open(f'losses-{name}.json', 'w') as f:
        json.dump(model_losses[best_model_idx], f)


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    subj = sys.argv[1].zfill(3)
    roi = sys.argv[2]
    exp = sys.argv[3] if (sys.argv[3] == "PLCB" or sys.argv[3] == "LSD") else "PLCB"
    name = f"sub-{subj}-{exp}-ROI{roi}"
    fname = f"{name}.txt"
    print(fname)
    observed_bold = np.loadtxt('./time_series/'+fname , delimiter=',')
    f2, observed_csd = csd(observed_bold, observed_bold, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)
    train_multiple(observed_csd, name)