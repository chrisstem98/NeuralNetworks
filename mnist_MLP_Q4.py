# -*- coding: utf-8 -*-
# Q4: New MLP with 2 hidden layers + Dropout + Confusion Matrix + CSV logging

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from logger import log_results   # CSV Logging

# GPU check
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
input_size = 784
hidden1 = 500
hidden2 = 200
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
dropout_p = 0.2     # Dropout for Q4

# Dataset
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ================================================================
#        NEW MLP MODEL FOR Q4 (2 hidden layers + dropout)
# ================================================================
class MLP_Q4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = MLP_Q4().to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ================================================================
#                   TRAINING LOOP (Q1 style)
# ================================================================
train_start = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] avg loss: {running_loss/len(train_loader):.4f}")

train_time = time.time() - train_start
print(f"Training time: {train_time:.2f} seconds")


# ================================================================
#                       TESTING LOOP
# ================================================================
test_start = time.time()

model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []
mis_images = []
mis_pred = []
mis_true = []

with torch.no_grad():
    for images, labels in test_loader:
        flat = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(flat)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # store predictions
        pred_cpu = predicted.cpu()
        labels_cpu = labels.cpu()

        all_preds.extend(pred_cpu.numpy())
        all_labels.extend(labels_cpu.numpy())

        mism = (pred_cpu != labels_cpu)
        if mism.any():
            mis_images.extend(images.cpu()[mism])
            mis_pred.extend(pred_cpu[mism])
            mis_true.extend(labels_cpu[mism])

test_time = time.time() - test_start
acc = 100.0 * correct / total

print(f"Test time: {test_time:.2f} seconds")
print(f"Accuracy: {acc:.2f}%")


# ================================================================
#            CSV LOGGING (Format A) → results_MLP2_Q4.csv
# ================================================================
log_results(
    "results_MLP2_Q4.csv",
    {
        "model": "MLP2_Q4",
        "run": 1,   # Updated externally in Colab loop
        "accuracy": acc,
        "train_time": train_time,
        "test_time": test_time
    }
)


# ================================================================
#             CONFUSION MATRIX (Saved PNG only)
# ================================================================
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MLP Q4")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/cm_MLP_Q4.png", dpi=200)
plt.close()


# ================================================================
#       MISCLASSIFIED EXAMPLES (Saved PNG only)
# ================================================================
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4,4,i+1)
    plt.imshow(mis_images[i].squeeze(), cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.suptitle("Misclassified - MLP Q4")
plt.savefig("results/misclassified_MLP_Q4.png", dpi=200)
plt.close()