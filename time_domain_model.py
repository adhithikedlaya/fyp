import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Euler_1stOrderForward import getEulerBOLD

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor, y_train_tensor = getEulerBOLD()
y_train_tensor = torch.tensor(y_train_tensor)

class TimeDomainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # assuming we are fitting noise parameters for now? - how would multiple regions work?
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        #GET BOLD SIGNAL from PDCM WITH NOISE ADDED
        _, yhat = getEulerBOLD(self.alpha.item(), self.beta.item(), noise=True)
        return torch.tensor(yhat, requires_grad=True)

    
model = TimeDomainModel().to(device)

lr = 0.01
n_epochs = 1000

def complex_mse_loss(output, target):
     # CALC CSD LOSS VS SIMULATED SIGNAL
    loss =  (((output - target)**2).sum())
    print(output, target)
    return loss

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()
    yhat = model()
    loss = complex_mse_loss(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    


print(yhat)
