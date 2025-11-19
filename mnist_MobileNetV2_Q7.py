# -*- coding: utf-8 -*-
# Q7: MobileNetV2 PARTIAL Fine-Tuning (last layers unfrozen)
# Includes: Confusion Matrix, Misclassified Samples, CSV Logging

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

from logger import log_results    # CSV logging

# ============================================================
#                     DEVICE SETUP
# ============================================================
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
#                     HYPERPARAMETERS
# ============================================================
batch_size = 64
num_epochs = 5
learning_rate = 0.0005     # higher than full FT

# ============================================================
#                    DATA TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST("./data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================
#                PARTIALLY UNFROZEN MobileNetV2
# ============================================================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze everything by default
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layers for partial fine-tuning
for idx in [14, 15, 16, 17]:
    for param in model.features[idx].parameters():
        param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 10)
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================
#                      TRAINING LOOP
# ============================================================
train_start = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss = {running_loss/len(train_loader):.4f}")

train_time = time.time() - train_start
print(f"Training time: {train_time:.2f} seconds")

# ============================================================
#                        TESTING LOOP
# ============================================================
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
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

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
acc = 100 * correct / total

print(f"Test time: {test_time:.2f} seconds")
print(f"Accuracy: {acc:.2f}%")

# ============================================================
#                      CSV LOGGING
# ============================================================
log_results(
    "results_MobileNetV2_Q7.csv",
    {
        "model": "MobileNetV2_Q7_PARTIAL_FT",
        "run": 1,
        "accuracy": acc,
        "train_time": train_time,
        "test_time": test_time
    }
)

# ============================================================
#                  CONFUSION MATRIX SAVE
# ============================================================
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MobileNetV2 PARTIAL FT")
plt.savefig("results/cm_MobileNetV2_Q7.png", dpi=200)
plt.close()

# ============================================================
#               MISCLASSIFIED SAMPLES SAVE
# ============================================================
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4,4,i+1)
    # Convert 3-channel tensor to grayscale
    img = mis_images[i].permute(1, 2, 0).numpy()
    img = img[:,:,0]
    plt.imshow(img, cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.suptitle("Misclassified - MobileNetV2 PARTIAL FT")
plt.savefig("results/misclassified_MobileNetV2_Q7.png", dpi=200)
plt.close()

print("Saved: cm_MobileNetV2_Q7.png and misclassified_MobileNetV2_Q7.png")
