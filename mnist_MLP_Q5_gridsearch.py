# -*- coding: utf-8 -*-
# Q5: Grid Search for MLP with 1 hidden layer + dropout + validation

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import pandas as pd

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters (fixed by assignment)
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Grid search params
hidden_sizes = [200, 400, 600, 800, 1000]
dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# Dataset
dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)

# 90/10 train/val split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


# =============================
#   MODEL DEFINITION FOR Q5
# =============================
class MLP_Q5(nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


criterion = nn.NLLLoss()

results = []

# =============================
#     GRID SEARCH LOOP
# =============================
for h in hidden_sizes:
    for d in dropout_values:
        print(f"\nTesting hidden={h}, dropout={d}")

        model = MLP_Q5(h, d).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # TRAIN
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images = images.view(-1, 28*28).to(device)
                labels = labels.to(device)

                loss = criterion(model(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # VALIDATION ACC
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 28*28).to(device)
                labels = labels.to(device)

                _, predicted = torch.max(model(images),1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation acc = {val_acc:.2f}%")

        results.append({
            "hidden_size": h,
            "dropout": d,
            "val_accuracy": val_acc
        })


# Save results
df = pd.DataFrame(results)
df.to_csv("grid_search_results_MLP_Q5.csv", index=False)

print("\nGrid search complete! Saved to grid_search_results_MLP_Q5.csv")
print(df.sort_values("val_accuracy", ascending=False).head())
