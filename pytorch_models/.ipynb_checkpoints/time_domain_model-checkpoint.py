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

def get_torch_rand(min, max):
    return torch.rand(()) * (max - min) + min

class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.4))
        self.lamb = nn.Parameter(torch.tensor(0.2))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.psi = nn.Parameter(torch.tensor(0.6))
        self.phi = nn.Parameter(torch.tensor(1.5))
        self.chi = nn.Parameter(torch.tensor(0.6))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(sigma=self.sigma, mu=self.mu, lamb=self.lamb, alpha=1.0, beta=self.beta, noise=True, length=500, phi=self.phi, psi=self.psi, chi=self.chi)
        yhat = torch.stack(yhat)
        yhat = yhat / torch.std(yhat)
        return yhat


def complex_mse_loss(output, csdy):
    output = output / torch.std(output)
    f, csdx = csd(output, output, fs=100, nperseg=10000)
    mse_real = torch.mean(torch.abs(torch.real(csdx) - torch.real(csdy))**2)
    mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real + mse_imaginary
    return loss

def train(observed_csd, name):
    model = TimeDomainModel().to(device)
    lr = 0.01
    n_epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        yhat = model()
        loss = complex_mse_loss(yhat, observed_csd, f)
        print("Loss: ", loss)
        print("Params: ", model.sigma, model.mu, model.lamb, model.beta, model.phi, model.psi, model.chi)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        
    _, final_y = getEulerBOLD(sigma=model.sigma, mu=model.mu, lamb=model.lamb, beta=model.beta, phi=model.phi, psi=model.psi, chi=model.chi, noise=True, length=1000)
    final_y = final_y / torch.std(final_y)
    f, csdx = csd(final_y, final_y, fs=100, nperseg=10000)
    np.savetxt(f"final_spectrum_{name}.txt", csdx.detach().numpy(), delimiter=',')

    # Save losses array as JSON
    with open(f'losses-{name}.json', 'w') as f:
        json.dump(losses, f)

def train_multiple(observed_csd, name):
    lr = 0.01
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    num_initializations = 5
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
            print("Params: ", model.sigma, model.mu, model.lamb, model.beta, model.psi, model.phi, model.chi)
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
    observed_bold = np.loadtxt('/rds/general/user/ak1920/home/fyp/fyp/time_series/'+fname , delimiter=',')
    observed_bold = torch.from_numpy(observed_bold).to(device)
    f2, observed_csd = csd(observed_bold, observed_bold, fs=0.5, nperseg=64)
    observed_csd = observed_csd / torch.sum(observed_csd)
    start_time = time.perf_counter()
    train(observed_csd, name)
    end_time = time.perf_counter()
    print("Benchmarked!: ", end_time - start_time)
