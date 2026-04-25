# 🐧 Penguin Classification: Perceptron & Adaline Models

This project implements two fundamental neural network building blocks—**Perceptron** and **Adaline** (Adaptive Linear Neuron)—from scratch using `NumPy`. The models are applied to the "Palmer Penguins" dataset to perform binary classification based on physical features.

---

## 📌 Project Overview

**Data Preprocessing**: Handling missing values using group-wise imputation, categorical encoding, and feature scaling.
**Perceptron Model**: A classic binary classifier that updates weights based on classification error using a signum activation function.
**Adaline Model**: An improvement that uses a linear activation function and minimizes Mean Squared Error (MSE) via gradient descent.

---

## 🛠️ Workflow & Preprocessing

The `preprocess_data` function ensures high-quality input for the models:
* **Missing Value Imputation**: Fills nulls in features like `CulmenLength` and `FlipperLength` using the **Mean** specific to each species.
* **Feature Engineering**: Maps `OriginLocation` to integers and converts species labels to binary targets ($1$ and $-1$).
* **Standardization**: Applies `StandardScaler` to normalize feature scales, which is critical for Adaline’s convergence and gradient descent stability.

---

## 📊 Performance Analysis (From Project Report)

Based on the model training and testing, here are the key findings:

### ✅ Best Feature Combinations (100% Accuracy)
* **For Perceptron**: `Culmen Length` & `Flipper Length` (or `Culmen Depth`). These provide a "best-case scenario" where species are linearly separable with a wide margin.
* **For Adaline**: `Body Mass` & `Culmen Length` (or `Flipper Length` & `Body Mass`). These combinations allow the cost function to minimize error and establish an ideal linear separator.

### ⚠️ Failure Cases (Overlap Issues)
**Linear Inseparability**: Some features like `Culmen Depth` vs. `Body Mass` for certain species show heavy overlap.
**Observation**: Because the data is not linearly separable in these cases, the Perceptron cannot find an effective boundary, resulting in a significant drop in accuracy (e.g., 60%).

---

## 📈 Visualizing Decisions

| Scenario | Model | Feature 1 | Feature 2 | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Ideal Separation** | Perceptron | Culmen Length | Flipper Length | 100.00%  |
| **High Power** | Perceptron | Culmen Length | Culmen Depth | 100.00%  |
| **Strong Performance** | Adaline | Flipper Length | Body Mass | 100.00%  |
| **Failure Case** | Perceptron | Culmen Depth | Body Mass | 60.00%  |

---
