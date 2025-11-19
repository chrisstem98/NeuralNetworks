import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
gpu_mlp = pd.read_csv("NeuralNetworks/Results/Q1/results_MLP_Q1.csv")
cpu_mlp = pd.read_csv("NeuralNetworks/Results/Q2/results_MLP_CPU_Q2.csv")

gpu_cnn = pd.read_csv("NeuralNetworks/Results/Q1/results_CNN_Q1.csv")
cpu_cnn = pd.read_csv("NeuralNetworks/Results/Q2/results_CNN_CPU_Q2.csv")   
  
gpu_mob = pd.read_csv("NeuralNetworks/Results/Q1/results_MobileNetV2_Q1.csv")
cpu_mob = pd.read_csv("NeuralNetworks/Results/Q2/results_MobileNetV2_CPU_Q2.csv")

# Calculate means
gpu_train_times = [
    gpu_mlp["train_time"].mean(),
    gpu_cnn["train_time"].mean(),
    gpu_mob["train_time"].mean(),
]

cpu_train_times = [
    cpu_mlp["train_time"][0],
    cpu_cnn["train_time"][0],
    cpu_mob["train_time"][0],
]

gpu_test_times = [
    gpu_mlp["test_time"].mean(),
    gpu_cnn["test_time"].mean(),
    gpu_mob["test_time"].mean(),
]

cpu_test_times = [
    cpu_mlp["test_time"][0],
    cpu_cnn["test_time"][0],
    cpu_mob["test_time"][0],
]

gpu_acc = [
    gpu_mlp["accuracy"].mean(),
    gpu_cnn["accuracy"].mean(),
    gpu_mob["accuracy"].mean(),
]

cpu_acc = [
    cpu_mlp["accuracy"][0],
    cpu_cnn["accuracy"][0],
    cpu_mob["accuracy"][0],
]

models = ["MLP", "CNN", "MobileNetV2"]

# --------------------------- PLOT 1: TRAIN TIME ---------------------------
plt.figure(figsize=(10,6))
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, gpu_train_times, width, label="GPU")
plt.bar(x + width/2, cpu_train_times, width, label="CPU")

plt.ylabel("Seconds")
plt.title("Training Time: GPU vs CPU")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# --------------------------- PLOT 2: TEST TIME ---------------------------
plt.figure(figsize=(10,6))
plt.bar(x - width/2, gpu_test_times, width, label="GPU")
plt.bar(x + width/2, cpu_test_times, width, label="CPU")

plt.ylabel("Seconds")
plt.title("Testing Time: GPU vs CPU")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# --------------------------- PLOT 3: ACCURACY ---------------------------
plt.figure(figsize=(10,6))
plt.bar(x - width/2, gpu_acc, width, label="GPU")
plt.bar(x + width/2, cpu_acc, width, label="CPU")

plt.ylabel("Accuracy (%)")
plt.title("Accuracy: GPU vs CPU")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
