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
  
# x_train_tensor, y_train_tensor = getEulerBOLD(noise=True, length=500)
# print("got here")
# f2, csdy = csd(y_train_tensor[::200][-217:], y_train_tensor[::200][-217:], fs=0.5, nperseg=64)
# print("got here 2")
# csdy = csdy / torch.std(csdy)
# f2, csdy = csd(target, target, fs=0.5, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=200)
# y_train_tensor = np.loadtxt('./time_series/sub-001-PLCB-ROI0.txt', delimiter=',')

def get_torch_rand(min, max):
    return torch.rand(()) * (max - min) + min


class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(0.9))
        self.mu = nn.Parameter(torch.tensor(0.8))
        self.lamb = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.psi = nn.Parameter(torch.tensor(0.5))
        self.phi = nn.Parameter(torch.tensor(2.0))
        self.chi = nn.Parameter(torch.tensor(0.3))


    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(sigma=self.sigma, mu=self.mu, lamb=self.lamb, alpha=1.0, beta=self.beta, noise=True, length=1000, phi=self.phi, psi=self.psi, chi=self.chi)
        yhat = torch.stack(yhat)
        return yhat
        # return torch.tensor(yhat, requires_grad=True)

    



def complex_mse_loss(output, csdy, fo):
#     f1, csdx = csd(output, output, fs=100, noverlap=None,  window='hamming', scaling='density', nfft=4096, nperseg=1000)
    output = output / torch.std(output)
    f, csdx = csd(output, output, fs=100, nperseg=20000)

    mask_f = torch.isin(f, fo)
    csdx_ds = csdx[mask_f]
    
    # Define the frequency range
    min_freq = 0
    max_freq = 0.1

    # Get the indices of the frequencies within the desired range
    freq_indices = (fo >= min_freq) & (fo <= max_freq)

    # Extract the corresponding CSDs
    csdx_ds = csdx_ds[freq_indices]
    csdy = csdy[freq_indices]
    
    mse_real = torch.mean(torch.abs(torch.real(csdx_ds) - torch.real(csdy))**2)
#     mse_imaginary = torch.mean(torch.abs(torch.imag(csdx) - torch.imag(csdy))**2)
    loss = mse_real

    return loss


def train(observed_csd, f, name):
    model = TimeDomainModel().to(device)
    lr = 0.01
    n_epochs = 150
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

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
    final_y = torch.stack(final_y)
    final_y = final_y / torch.std(final_y)
    f, csdx = csd(final_y, final_y, fs=100, nperseg=20000)
    np.savetxt(f"final_spectrum_take_2_{name}.txt", csdx.detach().numpy(), delimiter=',')
        

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
    observed_bold = np.loadtxt('/rds/general/user/ak1920/home/fyp/fyp/time_series_take_2/'+fname , delimiter=',')
    observed_bold = torch.from_numpy(observed_bold).to(device)
    observed_bold = observed_bold / torch.std(observed_bold)
    f2, observed_csd = csd(observed_bold, observed_bold, fs=0.5, nperseg=100)
    start_time = time.perf_counter()
    train(observed_csd, f2, name)
    end_time = time.perf_counter()
    print("Benchmarked!: ", end_time - start_time)
