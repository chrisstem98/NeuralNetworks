import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Q6 results
df = pd.read_csv("NeuralNetworks/Results/Q6/results_MobileNetV2_Q6.csv")

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

print("\n====== Q6 SUMMARY TABLE ======\n")
print(summary.to_string(index=False))

summary.to_csv("Q6_summary_table.csv", index=False)
print("\nSaved: Q6_summary_table.csv")

# ---------------------------------------------------------
# --------------------- PLOTS -----------------------------
# ---------------------------------------------------------

# ---------- TRAIN TIME ----------
plt.figure(figsize=(8,5))
plt.bar(["MobileNetV2-Q6"], [train_mean], yerr=[train_std], capsize=6)
plt.title("Q6: Training Time (mean ± std)")
plt.ylabel("Seconds")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q6_train_time.png", dpi=200)
plt.show()

# ---------- TEST TIME ----------
plt.figure(figsize=(8,5))
plt.bar(["MobileNetV2-Q6"], [test_mean], yerr=[test_std], capsize=6)
plt.title("Q6: Testing Time (mean ± std)")
plt.ylabel("Seconds")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q6_test_time.png", dpi=200)
plt.show()

# ---------- ACCURACY ----------
plt.figure(figsize=(8,5))
plt.bar(["MobileNetV2-Q6"], [acc_mean], yerr=[acc_std], capsize=6)
plt.title("Q6: Accuracy (mean ± std)")
plt.ylabel("Accuracy (%)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q6_accuracy.png", dpi=200)
plt.show()

print("Saved plots: Q6_train_time.png, Q6_test_time.png, Q6_accuracy.png")
