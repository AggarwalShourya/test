import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class StandardNetwork(nn.Module):
    def __init__(self):
        super(StandardNetwork, self).__init__()
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

# Generate input data
x = torch.linspace(-9, 9, 1000)[:, None].float()
y_true = x**2 + x  # True solution (y(x) = x^2 + x)

# Define and train the model
model = StandardNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train_standard(model, x, y_true, optimizer, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

train_standard(model, x, y_true, optimizer)
