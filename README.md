# 🐧 Penguin Classification: Perceptron & Adaline Models

This project implements two fundamental neural network building blocks—**Perceptron** and **Adaline** (Adaptive Linear Neuron)—from scratch using `NumPy`. The models are applied to the "Palmer Penguins" dataset to perform binary classification between species based on physical features.

---

## 📌 Project Overview

* **Data Preprocessing**: Handling missing values using group-wise imputation, categorical encoding, and feature scaling.
* **Perceptron Model**: A classic binary classifier that updates weights based on classification error using a signum activation function.
* **Adaline Model**: An improvement over the Perceptron that uses a linear activation function for weight updates, minimizing Mean Squared Error (MSE) via gradient descent.

---

## 🛠️ Workflow

### 1. Preprocessing Data
The `preprocess_data` function ensures the data is clean and ready for training:
* **Missing Values**: Fills nulls in `CulmenLength`, `CulmenDepth`, etc., using the **Mean** of that specific species.
* **Encoding**: Maps `OriginLocation` to integers and converts species labels to binary targets ($1$ and $-1$).
* **Splitting**: Uses a 60/40 train-test split with **stratification** to maintain class balance.
* **Standardization**: Applies `StandardScaler` to normalize feature scales, which is critical for Adaline’s convergence.

### 2. Implementation Details

#### 🧠 Perceptron Model
The Perceptron updates its weights only when it misclassifies a sample.
* **Activation**: $f(v) = 1$ if $v \geq 0$ else $-1$.
* **Weight Update Rule**: $w = w + \eta \cdot (target - prediction) \cdot x$

#### 📉 Adaline Model
Adaline uses the "Delta Rule" to minimize the cost function (MSE).
* **Activation**: Linear during training; Threshold (Signum) during testing.
* **Convergence**: Includes an `mse_threshold`. If the error falls below this limit, training stops early.

---

## 📊 Comparison Table

| Feature | Perceptron | Adaline |
| :--- | :--- | :--- |
| **Learning Rule** | Based on output error (Hard limit) | Based on cost function (Linear error) |
| **Stability** | May fluctuate if data is not linearly separable | Converges to the least-squares solution |
| **Stopping Criteria** | Fixed number of Epochs | Epochs or MSE Threshold |

---

**Developed with ❤️ by [Your Name]**
