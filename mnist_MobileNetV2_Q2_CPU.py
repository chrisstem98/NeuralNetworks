# -*- coding: utf-8 -*-
# Q2: MobileNetV2 CPU-only version

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from logger import log_results

# Force CPU even if GPU exists
device = torch.device("cpu")
print("Using CPU only")

# Hyperparameters
batch_size = 100
num_epochs = 3
learning_rate = 0.001

########################################################
# DATA TRANSFORMS (correct for MobileNetV2 pretrained)
########################################################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

########################################################
# Load pretrained MobileNetV2
########################################################
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

########################################################
# TRAINING LOOP (CPU ONLY)
########################################################
train_start = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {running_loss/len(train_loader):.4f}")

train_time = time.time() - train_start
print(f"CPU Training time: {train_time:.2f} seconds")

########################################################
# TESTING LOOP (CPU ONLY)
########################################################
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
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # logging predictions
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
print(f"CPU Test time: {test_time:.2f} seconds")

acc = 100.0 * correct / total
print(f"CPU Accuracy: {acc:.2f}%")

############################################################
# CSV LOGGING (Format A)
############################################################
log_results(
    "results_MobileNetV2_CPU.csv",
    {
        "model": "MobileNetV2_CPU",
        "run": 1,  # Colab loop will update
        "accuracy": acc,
        "train_time": train_time,
        "test_time": test_time
    }
)

############################################################
# CONFUSION MATRIX (Saved PNG Only)
############################################################
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MobileNetV2 CPU")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/cm_MobileNetV2_CPU.png", dpi=200)
plt.close()

############################################################
# MISCLASSIFIED VISUALIZATION (Saved PNG Only)
############################################################
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4,4,i+1)
    img = mis_images[i].permute(1,2,0).numpy()
    img = img[:,:,0]
    plt.imshow(img, cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.suptitle("Misclassified Samples - MobileNetV2 CPU")
plt.savefig("results/misclassified_MobileNetV2_CPU.png", dpi=200)
plt.close()
