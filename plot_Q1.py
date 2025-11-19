# plot_Q1.py
# Loads Q1 CSVs, computes mean/std tables, saves summary CSV, and generates plots

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Load CSVs for Q1 (GPU baseline)
# ---------------------------------------------------------
mlp = pd.read_csv("NeuralNetworks/Results/Q1/results_MLP_Q1.csv")
cnn = pd.read_csv("NeuralNetworks/Results/Q1/results_CNN_Q1.csv")
mob = pd.read_csv("NeuralNetworks/Results/Q1/results_MobileNetV2_Q1.csv")

models = ["MLP", "CNN", "MobileNetV2"]
csv_data = [mlp, cnn, mob]


# ---------------------------------------------------------
# Helper: compute mean/std per column
# ---------------------------------------------------------
train_means = [df["train_time"].mean() for df in csv_data]
train_stds  = [df["train_time"].std()  for df in csv_data]

test_means  = [df["test_time"].mean() for df in csv_data]
test_stds   = [df["test_time"].std()  for df in csv_data]

acc_means   = [df["accuracy"].mean() for df in csv_data]
acc_stds    = [df["accuracy"].std()  for df in csv_data]


# ---------------------------------------------------------
# Create summary table
# ---------------------------------------------------------
summary = pd.DataFrame({
    "Model": models,
    "Train Time Mean (s)": train_means,
    "Train Time Std (s)": train_stds,
    "Test Time Mean (s)": test_means,
    "Test Time Std (s)": test_stds,
    "Accuracy Mean (%)": acc_means,
    "Accuracy Std (%)": acc_stds
})

print("\n=== Q1 SUMMARY TABLE ===")
print(summary)

# Save summary table to CSV
summary.to_csv("Q1_summary_table.csv", index=False)
print("\nSaved Q1_summary_table.csv\n")


# ---------------------------------------------------------
# --------------------- PLOTS -----------------------------
# ---------------------------------------------------------

# ---------- Plot 1: TRAIN TIME ----------
plt.figure(figsize=(10,5))
plt.bar(models, train_means, yerr=train_stds, capsize=6)
plt.ylabel("Seconds")
plt.title("Q1: Training Time (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q1_train_time.png", dpi=200)
plt.show()


# ---------- Plot 2: TEST TIME ----------
plt.figure(figsize=(10,5))
plt.bar(models, test_means, yerr=test_stds, capsize=6)
plt.ylabel("Seconds")
plt.title("Q1: Testing Time (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q1_test_time.png", dpi=200)
plt.show()


# ---------- Plot 3: ACCURACY ----------
plt.figure(figsize=(10,5))
plt.bar(models, acc_means, yerr=acc_stds, capsize=6)
plt.ylabel("Accuracy (%)")
plt.title("Q1: Accuracy (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q1_accuracy.png", dpi=200)
plt.show()


print("Plots saved: Q1_train_time.png, Q1_test_time.png, Q1_accuracy.png")