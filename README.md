# 🐧 Penguin Classification: Perceptron & Adaline Models

This project implements two fundamental neural network building blocks—**Perceptron** and **Adaline** (Adaptive Linear Neuron)—from scratch using `NumPy`. The models are applied to the "Palmer Penguins" dataset to perform binary classification based on physical features.

---

## 📌 Project Overview

* [cite_start]**Data Preprocessing**: Handling missing values using group-wise imputation, categorical encoding, and feature scaling[cite: 38].
* [cite_start]**Perceptron Model**: A classic binary classifier that updates weights based on classification error using a signum activation function[cite: 1, 12].
* [cite_start]**Adaline Model**: An improvement that uses a linear activation function and minimizes Mean Squared Error (MSE) via gradient descent[cite: 20].

---

## 🛠️ Workflow & Preprocessing

The `preprocess_data` function ensures high-quality input for the models:
* **Missing Value Imputation**: Fills nulls in features like `CulmenLength` and `FlipperLength` using the **Mean** specific to each species.
* **Feature Engineering**: Maps `OriginLocation` to integers and converts species labels to binary targets ($1$ and $-1$).
* **Standardization**: Applies `StandardScaler` to normalize feature scales, which is critical for Adaline’s convergence and gradient descent stability.

---

## 📊 Performance Analysis (From Project Report)

[cite_start]Based on the model training and testing, here are the key findings[cite: 37]:

### ✅ Best Feature Combinations (100% Accuracy)
* **For Perceptron**: `Culmen Length` & `Flipper Length` (or `Culmen Depth`). [cite_start]These provide a "best-case scenario" where species are linearly separable with a wide margin[cite: 38, 39].
* **For Adaline**: `Body Mass` & `Culmen Length` (or `Flipper Length` & `Body Mass`). [cite_start]These combinations allow the cost function to minimize error and establish an ideal linear separator[cite: 41, 43].

### ⚠️ Failure Cases (Overlap Issues)
* [cite_start]**Linear Inseparability**: Some features like `Culmen Depth` vs. `Body Mass` for certain species show heavy overlap[cite: 16].
* [cite_start]**Observation**: Because the data is not linearly separable in these cases, the Perceptron cannot find an effective boundary, resulting in a significant drop in accuracy (e.g., 60%)[cite: 15, 17].

---

## 📈 Visualizing Decisions

| Scenario | Model | Feature 1 | Feature 2 | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Ideal Separation** | Perceptron | Culmen Length | Flipper Length | [cite_start]100.00% [cite: 1] |
| **High Power** | Perceptron | Culmen Length | Culmen Depth | [cite_start]100.00% [cite: 10] |
| **Strong Performance** | Adaline | Flipper Length | Body Mass | [cite_start]100.00% [cite: 19] |
| **Failure Case** | Perceptron | Culmen Depth | Body Mass | [cite_start]60.00% [cite: 15] |

---
