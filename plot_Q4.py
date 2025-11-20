import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Q4 results
df = pd.read_csv("NeuralNetworks/Results/Q4/results_MLP2_Q4.csv")

# ---------------------------------------------------------
# CALCULATE STATISTICS
# ---------------------------------------------------------

train_mean = df["train_time"].mean()
train_std  = df["train_time"].std()

test_mean  = df["test_time"].mean()
test_std   = df["test_time"].std()

acc_mean   = df["accuracy"].mean()
acc_std    = df["accuracy"].std()

# ---------------------------------------------------------
# CREATE SUMMARY TABLE
# ---------------------------------------------------------

summary = pd.DataFrame({
    "Metric": ["Train Time (s)", "Test Time (s)", "Accuracy (%)"],
    "Mean": [train_mean, test_mean, acc_mean],
    "Std": [train_std, test_std, acc_std]
})

print("\n====== Q4 SUMMARY TABLE ======\n")
print(summary.to_string(index=False))

summary.to_csv("Q4_summary_table.csv", index=False)
print("\nSaved: Q4_summary_table.csv")

# ---------------------------------------------------------
# --------------------- PLOTS -----------------------------
# ---------------------------------------------------------

# ---------- TRAIN TIME ----------
plt.figure(figsize=(8,5))
plt.bar(["MLP-Q4"], [train_mean], yerr=[train_std], capsize=6)
plt.title("Q4: Training Time (mean ± std)")
plt.ylabel("Seconds")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q4_train_time.png", dpi=200)
plt.show()

# ---------- TEST TIME ----------
plt.figure(figsize=(8,5))
plt.bar(["MLP-Q4"], [test_mean], yerr=[test_std], capsize=6)
plt.title("Q4: Testing Time (mean ± std)")
plt.ylabel("Seconds")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q4_test_time.png", dpi=200)
plt.show()

# ---------- ACCURACY ----------
plt.figure(figsize=(8,5))
plt.bar(["MLP-Q4"], [acc_mean], yerr=[acc_std], capsize=6)
plt.title("Q4: Accuracy (mean ± std)")
plt.ylabel("Accuracy (%)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q4_accuracy.png", dpi=200)
plt.show()

print("Saved plots: Q4_train_time.png, Q4_test_time.png, Q4_accuracy.png")
