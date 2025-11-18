# -*- coding: utf-8 -*-
# Q5: Retrain best model from grid search on full training set + test evaluation
# Includes: CSV logging, Confusion Matrix, MISCLASSIFIED images

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from logger import log_results
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load best combination from grid search
df = pd.read_csv("grid_search_results_MLP_Q5.csv")
best = df.sort_values(["val_accuracy", "hidden_size"], ascending=[False, True]).iloc[0]
hidden_size = int(best["hidden_size"])
dropout_p = float(best["dropout"])



# Dataset
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# ================================================================
#          Best Model Definition (Using best hyperparameters)
# ================================================================
class MLP_Q5(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = MLP_Q5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# ================================================================
#                       TRAINING (FULL DATA)
# ================================================================
train_start = time.time()

for epoch in range(5):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.view(-1,28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5] loss={running_loss/len(train_loader):.4f}")

train_time = time.time() - train_start
print(f"Training time: {train_time:.2f} seconds")

# ================================================================
#                       TESTING
# ================================================================
test_start = time.time()
correct,total = 0,0

all_preds = []
all_labels = []

# For misclassified samples
mis_images = []
mis_pred = []
mis_true = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        flat = images.view(-1,28*28).to(device)
        labels = labels.to(device)

        outputs = model(flat)
        _, predicted = torch.max(outputs,1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # For confusion matrix
        pred_cpu = predicted.cpu()
        labels_cpu = labels.cpu()
        all_preds.extend(pred_cpu.numpy())
        all_labels.extend(labels_cpu.numpy())

        # MISCLASSIFIED
        mism = (pred_cpu != labels_cpu)
        if mism.any():
            mis_images.extend(images.cpu()[mism])
            mis_pred.extend(pred_cpu[mism])
            mis_true.extend(labels_cpu[mism])

test_time = time.time() - test_start
acc = 100 * correct / total

print(f"Test time: {test_time:.2f} seconds")
print(f"Accuracy: {acc:.2f}%")

# ================================================================
#                     CSV LOGGING
# ================================================================
log_results(
    "results_MLP_Q5.csv",
    {
        "model": "MLP_Q5",
        "run": 1,
        "accuracy": acc,
        "train_time": train_time,
        "test_time": test_time
    }
)

# ================================================================
#                     CONFUSION MATRIX
# ================================================================
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MLP Q5")
plt.savefig("results/cm_MLP_Q5.png", dpi=200)
plt.close()

# ================================================================
#                 MISCLASSIFIED VISUALIZATION
# ================================================================
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4,4,i+1)
    plt.imshow(mis_images[i].squeeze(), cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.suptitle("Misclassified Samples - MLP Q5")
plt.savefig("results/misclassified_MLP_Q5.png", dpi=200)
plt.close()

print("Saved: cm_MLP_Q5.png and misclassified_MLP_Q5.png")
