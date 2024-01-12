import torch
import torch.optim as optim
import torch.nn as nn
from csd_calculation import csd, f
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#TODO make sure format of observed data matches output of forward
# test the ML
# plot the observed and output of the ML against each other uses dotted lines



x_train_tensor = torch.from_numpy(f).float().to(device) #input data - what does our model actually take in - omega - the input frequencies
y_train_tensor = torch.complex(torch.from_numpy(csd.real), torch.from_numpy(csd.imag)).to(device) # this comes from csd var..  

print("sd", device)

#ASSUMPTION endogenous fluctuations = observation noise
class ManualLinearRegression(nn.Module):
    def __init__(self, num_regions):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.alphas_e = nn.Parameter(torch.rand(num_regions, requires_grad=True, dtype=torch.float))
        self.betas_e = nn.Parameter(torch.rand(num_regions, requires_grad=True, dtype=torch.float)) #0.5 - 3.5
        # self.alpha = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.alphas_v = nn.Parameter(torch.rand(num_regions, requires_grad=True, dtype=torch.float))
        self.betas_v = nn.Parameter(torch.rand(num_regions, requires_grad=True, dtype=torch.float))
        # self.A = torch.tensor([[-1/2, 0], [0, -1/2]])
        self.A = nn.Parameter(torch.randn(num_regions, num_regions, requires_grad=True, dtype=torch.float))
        self.num_regions = num_regions
        
    def h(self, omega):
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        hrf = (6 * (i * omega + 1) ** 10 - 1) / (5 * (i * omega + 1) ** 16)
        return hrf * torch.eye(self.num_regions)
    
    def g(self, omega, alphas, betas):
        # print(alphas, torch.full([self.num_regions], omega), (-1 * betas))
        # return (1/((omega + 1) ** 2)) * torch.eye(self.num_regions)
        return torch.diag(alphas * torch.full([self.num_regions], omega) ** (-1 * betas))

    def forward_single_freq(self, omega):
        # Computes the outputs / predictions
        I = torch.eye(self.num_regions)
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        hrf = self.h(omega)
        X = torch.linalg.inv(i * omega * I - self.A) 
        G_v = self.g(omega, self.alphas_v, self.betas_v) 
        G_e = self.g(omega, self.alphas_e, self.betas_e) 
        X_t = torch.linalg.inv(-1 * i * omega * I - self.A.T) 
        hrf_T = torch.conj(hrf).T

        X = X.to(torch.complex64)
        G_v = G_v.to(torch.complex64)
        X_t = X_t.to(torch.complex64)
        G_e = G_e.to(torch.complex64)
        hrf_T = hrf_T.to(torch.complex64)
        result =  hrf @ X @ G_v @ X_t @ hrf_T + G_e
        return result

    def forward(self, freqs):
        csd_curves = torch.empty((self.num_regions ** 2, freqs.size(0)), dtype=torch.complex64)
        for (i, freq) in enumerate(freqs):
            csds = self.forward_single_freq(freq).view(-1)
            for (reg, csd_val) in enumerate(csds):
                csd_curves[reg, i] = csd_val  
        return csd_curves

    
    def plot(self, gs):
        for g in gs:
            plt.plot(f, g.real)
            plt.plot(f, g.imag)
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.show()
        return
    
    
# torch.manual_seed(12)
# # Now we can create a model and send it at once to the device
model = ManualLinearRegression(2).to(device)

# model.A = torch.nn.Parameter(torch.tensor([[-1/2, 0], [2, -1/2]]))
# yhat = model.forward(x_train_tensor)


# # We can also inspect its parameters using its state_dict
# print(model.state_dict())

lr = 0.01
n_epochs = 1000

def complex_mse_loss(output, target):
    loss =  torch.abs(((output[3].real - target[3].real)**2).sum())
    return loss

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    # What is this?!?
    model.train()

    # No more manual prediction!
    # yhat = a + b * x_tensor
    yhat = model(x_train_tensor)
    loss = complex_mse_loss(yhat, y_train_tensor)
    
    loss.backward()
    print(loss) 
    optimizer.step()
    optimizer.zero_grad()
    


print(yhat)
# for i in range(4):

g = yhat[3].detach().numpy()
y = y_train_tensor[3].detach().numpy()
plt.plot(f, g)
plt.plot(f, y, label="target")
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
# y = y_train_tensor[2].detach().numpy()
# plt.plot(f[1:], y_train_tensor[1:].real)
#plt.plot(f, y, label="target")
#plt.plot(f, y.imag, label="target")
#plt.plot(f[1:], y_train_tensor[1:])
# plt.plot(f, g)
#plt.plot(f, g.imag)

# plt.legend()
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()