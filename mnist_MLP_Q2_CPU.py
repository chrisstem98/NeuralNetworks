# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from logger import log_results

print("Using CPU (Q2)")
device = torch.device("cpu")

# Hyperparameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Dataset
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = MLP().to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN
train_start = time.time()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_time = time.time() - train_start

# TEST
test_start = time.time()

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        _, predicted = torch.max(model(images), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_time = time.time() - test_start

acc = 100 * correct / total

print(f"CPU MLP: ACC={acc:.2f}%, TRAIN={train_time:.2f}s, TEST={test_time:.2f}s")

log_results(
    "results_MLP_CPU.csv",
    {"model": "MLP_CPU", "run": 1, "accuracy": acc, "train_time": train_time, "test_time": test_time}
)
