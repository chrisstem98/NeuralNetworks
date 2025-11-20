import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# LOAD ALL RESULTS
# ---------------------------------------------------------

paths = {
    "MLP_Q1": "NeuralNetworks/Results/Q1/results_MLP_Q1.csv",
    "CNN_Q1": "NeuralNetworks/Results/Q1/results_CNN_Q1.csv",
    "MobileNetV2_Q1": "NeuralNetworks/Results/Q1/results_MobileNetV2_Q1.csv",

    "MLP_CPU": "NeuralNetworks/Results/Q2/results_MLP_CPU_Q2.csv",
    "CNN_CPU": "NeuralNetworks/Results/Q2/results_CNN_CPU_Q2.csv",
    "MobileNetV2_CPU": "NeuralNetworks/Results/Q2/results_MobileNetV2_CPU_Q2.csv",
    
    "MLP_Q4": "NeuralNetworks/Results/Q4/results_MLP2_Q4.csv",
    "MLP_Q5": "NeuralNetworks/Results/Q5/results_MLP_Q5.csv",

    "MobileNetV2_Q6": "NeuralNetworks/Results/Q6/results_MobileNetV2_Q6.csv",
    "MobileNetV2_Q7": "NeuralNetworks/Results/Q7/results_MobileNetV2_Q7.csv"
}

dfs = {name: pd.read_csv(path) for name, path in paths.items()}

# ---------------------------------------------------------
# BUILD SUMMARY TABLE
# ---------------------------------------------------------
rows = []

for name, df in dfs.items():
    rows.append({
        "Model": name,
        "Train Mean": df["train_time"].mean(),
        "Train Std":  df["train_time"].std(),
        "Test Mean": df["test_time"].mean(),
        "Test Std":  df["test_time"].std(),
        "Acc Mean": df["accuracy"].mean(),
        "Acc Std": df["accuracy"].std()
    })

summary = pd.DataFrame(rows)
summary.to_csv("Q8_summary_table.csv", index=False)
print("\n===== Q8 SUMMARY TABLE =====\n")
print(summary.to_string(index=False))

# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

models = summary["Model"]
acc_means = summary["Acc Mean"]
acc_stds  = summary["Acc Std"]

train_means = summary["Train Mean"]
train_stds  = summary["Train Std"]

test_means = summary["Test Mean"]
test_stds  = summary["Test Std"]

# ---------- Accuracy ----------
plt.figure(figsize=(14,5))
plt.bar(models, acc_means, yerr=acc_stds, capsize=5)
plt.xticks(rotation=45)
plt.ylabel("Accuracy (%)")
plt.title("Q8: Accuracy Comparison (Q1–Q7)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Q8_accuracy.png", dpi=200)
plt.show()

# ---------- Train Time ----------
plt.figure(figsize=(14,5))
plt.bar(models, train_means, yerr=train_stds, capsize=5)
plt.xticks(rotation=45)
plt.ylabel("Seconds")
plt.title("Q8: Training Time Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Q8_train_time.png", dpi=200)
plt.show()

# ---------- Test Time ----------
plt.figure(figsize=(14,5))
plt.bar(models, test_means, yerr=test_stds, capsize=5)
plt.xticks(rotation=45)
plt.ylabel("Seconds")
plt.title("Q8: Testing Time Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Q8_test_time.png", dpi=200)
plt.show()

print("Saved plots: Q8_accuracy.png, Q8_train_time.png, Q8_test_time.png")
