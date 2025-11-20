import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. LOAD CSVs
# ---------------------------------------------------------
grid = pd.read_csv("NeuralNetworks/Results/Q5/grid_search_results_MLP_Q5.csv")
best_runs = pd.read_csv("NeuralNetworks/Results/Q5/results_MLP_Q5.csv")   # final best-model runs

# ---------------------------------------------------------
# 2. GRID SEARCH HEATMAP (val_accuracy vs hidden/dropout)
# ---------------------------------------------------------

# Pivot: rows = hidden_size, cols = dropout, values = val_accuracy
pivot = grid.pivot(index="hidden_size", columns="dropout", values="val_accuracy")

plt.figure(figsize=(8,6))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="Validation Accuracy (%)")
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.xlabel("Dropout p")
plt.ylabel("Hidden size")
plt.title("Q5: Grid Search Validation Accuracy")
# add text annotations
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white")
plt.tight_layout()
plt.savefig("Q5_grid_heatmap.png", dpi=200)
plt.show()

# ---------------------------------------------------------
# 3. BEST HYPERPARAMETERS FROM GRID
# ---------------------------------------------------------
best_row = grid.sort_values(["val_accuracy", "hidden_size"], ascending=[False, True]).iloc[0]
best_hidden = int(best_row["hidden_size"])
best_dropout = float(best_row["dropout"])
best_val_acc = best_row["val_accuracy"]

print("\n===== Q5 BEST HYPERPARAMETERS FROM GRID SEARCH =====")
print(f"Hidden size   : {best_hidden}")
print(f"Dropout p     : {best_dropout}")
print(f"Val accuracy  : {best_val_acc:.2f}%")

# Save best hyperparams to small CSV/text if you want
pd.DataFrame([{
    "hidden_size": best_hidden,
    "dropout": best_dropout,
    "val_accuracy": best_val_acc
}]).to_csv("Q5_best_hyperparams.csv", index=False)
print("Saved: Q5_best_hyperparams.csv")

# ---------------------------------------------------------
# 4. SUMMARY FOR BEST MODEL RUNS (results_MLP_Q5.csv)
# ---------------------------------------------------------

train_mean = best_runs["train_time"].mean()
train_std  = best_runs["train_time"].std()

test_mean  = best_runs["test_time"].mean()
test_std   = best_runs["test_time"].std()

acc_mean   = best_runs["accuracy"].mean()
acc_std    = best_runs["accuracy"].std()

summary = pd.DataFrame({
    "Metric": ["Train Time (s)", "Test Time (s)", "Accuracy (%)"],
    "Mean": [train_mean, test_mean, acc_mean],
    "Std": [train_std, test_std, acc_std]
})

print("\n===== Q5 BEST MODEL SUMMARY (FROM RUNS) =====\n")
print(summary.to_string(index=False))

summary.to_csv("Q5_summary_table.csv", index=False)
print("\nSaved: Q5_summary_table.csv")

# ---------------------------------------------------------
# 5. PLOTS FOR BEST MODEL
# ---------------------------------------------------------

# ---- TRAIN TIME ----
plt.figure(figsize=(7,5))
plt.bar(["MLP-Q5"], [train_mean], yerr=[train_std], capsize=6)
plt.ylabel("Seconds")
plt.title("Q5: Training Time (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q5_train_time.png", dpi=200)
plt.show()

# ---- TEST TIME ----
plt.figure(figsize=(7,5))
plt.bar(["MLP-Q5"], [test_mean], yerr=[test_std], capsize=6)
plt.ylabel("Seconds")
plt.title("Q5: Testing Time (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q5_test_time.png", dpi=200)
plt.show()

# ---- ACCURACY ----
plt.figure(figsize=(7,5))
plt.bar(["MLP-Q5"], [acc_mean], yerr=[acc_std], capsize=6)
plt.ylabel("Accuracy (%)")
plt.title("Q5: Accuracy (mean ± std)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Q5_accuracy.png", dpi=200)
plt.show()

print("Saved plots: Q5_grid_heatmap.png, Q5_train_time.png, Q5_test_time.png, Q5_accuracy.png")
