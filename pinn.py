import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 60)
        self.fc3 = nn.Linear(60, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigm(x)
        x = self.fc2(x)
        x = self.sigm(x)
        x = self.fc3(x)
        return x

model = network()

def loss(x):
    x.requires_grad = True
    y = model(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return torch.mean((dy_dx - (2 * x) - 1)**2) + (model(torch.tensor(0.0).unsqueeze(0).float()) - 0)**2

x = torch.linspace(-9, 9, 1000)[:, None].float()  # Ensure x is float32

optimizer = torch.optim.LBFGS(model.parameters())

def final():
    optimizer.zero_grad()
    criterion = loss(x)
    criterion.backward()
    return criterion

epochs = 20

for i in range(epochs):
    print(f"epoch:{i+1}")
    optimizer.step(final)
