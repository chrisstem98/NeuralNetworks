# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from logger import log_results

print("Using CPU (Q2)")
device = torch.device("cpu")

num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Dataset
train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return F.log_softmax(self.fc1(x), dim=1)


model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# TRAIN
train_start = time.time()

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_time = time.time() - train_start


# TEST + COLLECT PREDICTIONS
test_start = time.time()

correct, total = 0, 0
all_preds = []
all_labels = []
mis_images = []
mis_pred = []
mis_true = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Save misclassified samples
        for img, p, t in zip(images.cpu(), predicted.cpu(), labels.cpu()):
            if p != t:
                mis_images.append(img)
                mis_pred.append(int(p))
                mis_true.append(int(t))

test_time = time.time() - test_start
acc = 100 * correct / total


# ============= SAVE RESULTS FOLDER =============
os.makedirs("results", exist_ok=True)


# ============= CONFUSION MATRIX PNG =============
cm = confusion_matrix(all_labels, all_preds)

import matplotlib
matplotlib.use("Agg")  # για αποθήκευση χωρίς display

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - CNN CPU")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/cm_CNN_CPU.png", dpi=200)
plt.close()


# ============= MISCLASSIFIED IMG GRID =============
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4, 4, i+1)
    img = mis_images[i].squeeze().numpy()
    plt.imshow(img, cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.savefig("results/misclassified_CNN_CPU.png", dpi=200)
plt.close()


# PRINT
print(f"CPU CNN: ACC={acc:.2f}%, TRAIN={train_time:.2f}s, TEST={test_time:.2f}s")


# LOG RESULTS
log_results(
    "results_CNN_CPU.csv",
    {"model": "CNN_CPU", "run": 1, "accuracy": acc, "train_time": train_time, "test_time": test_time}
)
