# 🧠 Neural Networks – MNIST Experiments (Q1–Q8)

This repository contains all Python scripts, plots, and analysis for a full MNIST machine learning project using multiple neural network architectures and training strategies.

The project is organized into **eight questions (Q1–Q8)**, each focusing on a different aspect of model training, evaluation, and comparison.

---

## ✅ Q1 – Baseline Evaluation (GPU)
Train and evaluate three baseline models on GPU:
- **MLP**
- **CNN**
- **MobileNetV2** (classifier-only)

Collect:
- Training time
- Testing time
- Accuracy (5 runs per model)

Save:
- Confusion matrices  
- Misclassified samples

---

## ✅ Q2 – CPU vs GPU Comparison
Run each baseline model **once** on CPU only.

Compare:
- Training time
- Testing time
- Accuracy

against the GPU results from **Q1**.

---

## ✅ Q3 – Error Analysis
Using confusion matrices and misclassified images from **Q1**, analyze:
- The types of errors each model makes  
- Which digit pairs are most confusing  
- Which mistakes are common across models

---

## ✅ Q4 – Improved MLP Model
Create and train a modified MLP with:
- **Two hidden layers**
- **Dropout**

Run 5 experiments and compare results to the baseline MLP from **Q1**.

---

## ✅ Q5 – Grid Search (MLP Hyperparameters)
Perform grid search over:
- Hidden layer sizes  
- Dropout probabilities  

Procedure:
- Use a **90/10 train/validation** split  
- Train each configuration  
- Select the best-performing architecture  
- Retrain on full training set (5 runs)

---

## ✅ Q6 – MobileNetV2 Full Fine-Tuning
Unfreeze **all layers** of MobileNetV2 and fully fine-tune on MNIST.

Record:
- Training/testing time  
- Accuracy (5 runs)  

Save:
- Confusion matrix  
- Misclassified samples

---

## ✅ Q7 – MobileNetV2 Partial Fine-Tuning
Unfreeze only the **last few layers** of MobileNetV2 (partial FT).

Train for 5 runs and compare:
- Performance  
- Training time  
- Accuracy  

against Q1 and Q6.

---

## ✅ Q8 – Final Comparison of All Models
Aggregate all results from **Q1–Q7** into:
- A full comparison table  
- Accuracy plot  
- Training time plot  
- Testing time plot  

Conclude:
- Which model performs best  
- Cost vs performance tradeoffs  
- Final recommendation

---

## 📂 Project Structure (Suggested)

