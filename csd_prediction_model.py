import torch
import torch.optim as optim
import torch.nn as nn
from csd_calculation import csds, f
from csd_forward import simulated_csds
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, Output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(f).float().to(device)
y_train_tensor = torch.tensor(simulated_csds)

class Spectral_DCM_Model(nn.Module):
    def get_rand_in_range(self, a, b, square=False):
        if square:
            return a - b * torch.rand(self.num_regions, self.num_regions, requires_grad=True, dtype=torch.float) + b
        return a - b * torch.rand(self.num_regions, requires_grad=True, dtype=torch.float) + b
    
    def __init__(self, num_regions):
        super().__init__()
        self.num_regions = num_regions
        self.alphas_e = nn.Parameter(torch.tensor([0.1]))
        self.betas_e = nn.Parameter(torch.tensor([-0.1])) #0.5 - 3.5
        self.alphas_v = nn.Parameter(torch.tensor([-2.0]))
        self.betas_v = nn.Parameter(torch.tensor([-0.1]))
        self.A = nn.Parameter(torch.tensor([-2.0]))
        
    def h(self, omega):
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        hrf = (6 * (i * omega + 1) ** 10 - 1) / (5 * (i * omega + 1) ** 16)
        return hrf * torch.eye(self.num_regions)
    
    def g(self, omega, alphas, betas):
        # return (1/((omega + 1) ** 2)) * torch.eye(self.num_regions)
        return torch.diag(alphas * torch.full([self.num_regions], omega) ** (-1 * betas))

    # NOT BEING USED RN
    def make_self_connections_neg(self):
        a = self.A
        v = -torch.exp(torch.diag(self.A))
        mask = torch.diag(torch.ones_like(v))
        out = mask*torch.diag(v) + (1. - mask)*a
        return out

    def forward_single_freq(self, omega):
        A = self.A
        I = torch.eye(self.num_regions)
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        hrf = self.h(omega)
        hrf_T = torch.conj(hrf).T
        G_v = self.g(omega, self.alphas_v, self.betas_v) 
        G_v = G_v.to(torch.complex64)
        X = i * omega * I - A
        C = torch.linalg.solve(X, hrf, left=False)
        G_e = self.g(omega, self.alphas_e, self.betas_e) 
        result =  C @ G_v @ torch.conj(C).T + G_e

        return result

    def forward(self, freqs):
        csd_curves = torch.empty((self.num_regions ** 2, freqs.size(0)), dtype=torch.complex64)
        for (i, freq) in enumerate(freqs):
            csds = self.forward_single_freq(freq).view(-1)
            for (reg, csd_val) in enumerate(csds):
                csd_curves[reg, i] = csd_val  
        return csd_curves

    
    def plot(self, yhat, y_train_tensor):
        plt.autoscale(enable=True, axis='both', tight=None)
        print("A", self.A)
        print("alphas e", self.alphas_e)
        print("alphas v", self.alphas_v)
        print("betas e", self.betas_e)
        print("betas v", self.betas_v)
        for i in range(self.num_regions ** 2):
            g = yhat[i].detach().numpy()
            y = y_train_tensor[i].detach().numpy()
            plt.plot(f, g, label="simulated CSD")
            plt.plot(f, y, label="empirical CSD")
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.show()
    

def complex_mse_loss(output, target):
    loss =  ((output.real - target.real)**2).sum()
    return loss  

model = Spectral_DCM_Model(1).to(device)

lr = 0.01
n_epochs = 500

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()

    yhat = model(x_train_tensor)
    loss = complex_mse_loss(yhat, y_train_tensor)
    print(loss) 
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

model.plot(yhat, y_train_tensor)   