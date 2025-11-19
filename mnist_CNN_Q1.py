# MNIST CNN Classifier with Timing (Q1)
import time   # Added for Q1: timing training and testing

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Extra imports for confusion matrix + misclassified plots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from logger import log_results   # Added for CSV logging

# Confirm GPU availability
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001 

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transforms.ToTensor(),  
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# Define the neural network model
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

# Initialize the model
model = CNNet().to(device)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

###############################################################################
# TRAINING LOOP (Corrected + Added Timers for Q1)
###############################################################################
train_start = time.time()   # Added for Q1: start training timer

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # FIXED for CNN (no reshape)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] Average loss: {running_loss/len(train_loader):.4f}")

train_time = time.time() - train_start   # Added for Q1
print(f"Training time: {train_time:.2f} seconds")   # Added for Q1


###############################################################################
# TESTING LOOP (Corrected + Added Timers for Q1)
###############################################################################
test_start = time.time()   # Added for Q1: start testing timer

model.eval()
n_correct = 0
n_samples = 0

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

        # accuracy
        n_correct += (predicted == labels).sum().item()
        n_samples += labels.size(0)

        # store predictions
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # misclassified
        pred_cpu = predicted.cpu()
        labels_cpu = labels.cpu()
        mism = (pred_cpu != labels_cpu)

        if mism.any():
            mis_images.extend(images.cpu()[mism])
            mis_pred.extend(pred_cpu[mism])
            mis_true.extend(labels_cpu[mism])

test_time = time.time() - test_start   # Added for Q1
print(f"Test time: {test_time:.2f} seconds")   # Added for Q1

acc = 100.0 * n_correct / n_samples
print(f"Accuracy of the network on the 10000 test images: {acc:.2f} %")

###############################################################################
# CSV LOGGING
###############################################################################
log_results(
    "results_CNN.csv",
    {
        "model": "CNN",
        "run": 1,  # Updated externally in Colab loop
        "accuracy": acc,
        "train_time": train_time,
        "test_time": test_time
    }
)

###############################################################################
# CONFUSION MATRIX (Saved PNG Only)
###############################################################################
os.makedirs("results", exist_ok=True)
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/cm_CNN.png", dpi=200)
plt.close()

###############################################################################
# MISCLASSIFIED VISUALIZATION (Saved PNG Only)
###############################################################################
num_to_show = min(16, len(mis_images))
plt.figure(figsize=(10,10))

for i in range(num_to_show):
    plt.subplot(4,4,i+1)
    plt.imshow(mis_images[i].squeeze(), cmap="gray")
    plt.title(f"P:{mis_pred[i]} / T:{mis_true[i]}")
    plt.axis("off")

plt.suptitle("Misclassified Samples - CNN")
plt.savefig("results/misclassified_CNN.png", dpi=200)
plt.close()
