import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Custom imports
from models.slp import PerceptronModel
from models.adaline import AdalineModel
from utils.confusion_matrix import confusion_matrix
from preprocessing import preprocess_data

# Global variables
model = None
current_scale = None
location_encoder = None
canvas = None  # To hold the plot reference

def run_gui():
    window = tk.Tk()
    window.title("🐧 Penguins Advanced Classifier Pro")
    window.geometry("1300x750")  # Wider window for the integrated plot
    window.configure(bg="#f8f9fa")

    # --- Header ---
    header = tk.Label(window, text="Integrated Penguin Analysis Dashboard", 
                      font=("Helvetica", 18, "bold"), bg="#2c3e50", fg="white", pady=10)
    header.grid(row=0, column=0, columnspan=3, sticky="ew")

    # --- 1. Left Frame: Configuration ---
    settings_frame = tk.LabelFrame(window, text=" ⚙️ Configuration ", font=("Arial", 11, "bold"), 
                                   padx=15, pady=15, bg="white")
    settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    features = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"]
    classes = ["Adelie", "Gentoo", "Chinstrap"]

    tk.Label(settings_frame, text="Feature 1:", bg="white").grid(row=0, column=0, sticky="w")
    feat1_box = ttk.Combobox(settings_frame, values=features, width=20)
    feat1_box.grid(row=0, column=1, pady=5)

    tk.Label(settings_frame, text="Feature 2:", bg="white").grid(row=1, column=0, sticky="w")
    feat2_box = ttk.Combobox(settings_frame, values=features, width=20)
    feat2_box.grid(row=1, column=1, pady=5)

    tk.Label(settings_frame, text="Class 1:", bg="white").grid(row=2, column=0, sticky="w")
    c1_box = ttk.Combobox(settings_frame, values=classes, width=20)
    c1_box.grid(row=2, column=1, pady=5)

    tk.Label(settings_frame, text="Class 2:", bg="white").grid(row=3, column=0, sticky="w")
    c2_box = ttk.Combobox(settings_frame, values=classes, width=20)
    c2_box.grid(row=3, column=1, pady=5)

    # Hyperparameters
    lr_entry = tk.Entry(settings_frame, width=23); lr_entry.insert(0, "0.01")
    epoch_entry = tk.Entry(settings_frame, width=23); epoch_entry.insert(0, "100")
    
    tk.Label(settings_frame, text="LR:", bg="white").grid(row=4, column=0, sticky="w")
    lr_entry.grid(row=4, column=1, pady=5)
    tk.Label(settings_frame, text="Epochs:", bg="white").grid(row=5, column=0, sticky="w")
    epoch_entry.grid(row=5, column=1, pady=5)

    bias_var = tk.IntVar(value=1)
    tk.Checkbutton(settings_frame, text="Use Bias", variable=bias_var, bg="white").grid(row=6, column=1, sticky="w")

    algo_var = tk.StringVar(value="Perceptron")
    tk.Radiobutton(settings_frame, text="Perceptron", variable=algo_var, value="Perceptron", bg="white").grid(row=7, column=1, sticky="w")
    tk.Radiobutton(settings_frame, text="Adaline", variable=algo_var, value="Adaline", bg="white").grid(row=8, column=1, sticky="w")

    # --- 2. Middle Frame: Results & Predict ---
    middle_frame = tk.Frame(window, bg="#f8f9fa")
    middle_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    result_label = tk.Label(middle_frame, text="Results here...", font=("Courier", 10), bg="white", relief="sunken", height=8, width=35)
    result_label.pack(pady=5)

    predict_frame = tk.LabelFrame(middle_frame, text=" 🔍 Prediction ", bg="#eef2f7", padx=10, pady=10)
    predict_frame.pack(fill="x", pady=10)
    
    sample1 = tk.Entry(predict_frame, width=15); sample1.pack(pady=2)
    sample2 = tk.Entry(predict_frame, width=15); sample2.pack(pady=2)
    classify_result = tk.Label(predict_frame, text="Detected: --", font=("Arial", 12, "bold"), bg="#eef2f7")
    classify_result.pack(pady=5)

    # --- 3. Right Frame: The Integrated Plot Area ---
    plot_frame = tk.LabelFrame(window, text=" 📈 Visualization Area ", font=("Arial", 11, "bold"), bg="white")
    plot_frame.grid(row=1, column=2, padx=20, pady=10, sticky="nsew")

    # --- Embedded Plot Function ---
    def update_plot(model, X, y, f1, f2):
        global canvas
        if canvas: canvas.get_tk_widget().destroy() # Clear old plot

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
        ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', label='Class -1')

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_vals = np.linspace(x1_min, x1_max, 100)
        
        # Boundary Line math
        if not model.use_bias:
            x2_vals = -(model.w1 * x1_vals) / model.w2
        else:
            x2_vals = -(model.w1 * x1_vals + model.w0) / model.w2
            
        ax.plot(x1_vals, x2_vals, color='black', linewidth=2)
        ax.set_xlabel(f1); ax.set_ylabel(f2)
        ax.set_title("Decision Boundary")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Logic ---
    def train_model():
        global model, current_scale, location_encoder
        try:
            f1, f2 = feat1_box.get(), feat2_box.get()
            c1, c2 = c1_box.get(), c2_box.get()
            
            X_train, y_train, X_test, y_test, current_scale, location_encoder = preprocess_data(
                "D:/3rd year/2nd term/Neural Network/Penguin_Project/data/penguins.csv", f1, f2, c1, c2)

            if algo_var.get() == "Perceptron":
                model = PerceptronModel(float(lr_entry.get()), int(epoch_entry.get()), bool(bias_var.get()))
            else:
                model = AdalineModel(float(lr_entry.get()), int(epoch_entry.get()), bool(bias_var.get()), 0.001)

            model.train(X_train, y_train)
            acc, preds = model.test(X_test, y_test)
            tp, tn, fp, fn = confusion_matrix(y_test, preds)
            
            result_label.config(text=f"Accuracy: {acc:.2f}%\nTP: {tp} | FN: {fn}\nFP: {fp} | TN: {tn}")
            update_plot(model, X_train, y_train, f1, f2)
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

    def classify_sample():
        if not model: return
        try:
            # Classification logic same as before...
            v1, v2 = float(sample1.get()), float(sample2.get())
            sc = current_scale.transform([[v1, v2]])[0]
            net = (model.w0 * (1 if bias_var.get() else 0)) + (model.w1 * sc[0]) + (model.w2 * sc[1])
            res = c1_box.get() if net >= 0 else c2_box.get()
            classify_result.config(text=f"Result: {res}", fg="#27ae60")
        except: pass

    # Buttons
    tk.Button(settings_frame, text="🚀 TRAIN", command=train_model, bg="#27ae60", fg="white", font=("Arial", 10, "bold")).grid(row=10, column=0, columnspan=2, pady=20, sticky="ew")
    tk.Button(predict_frame, text="🎯 CLASSIFY", command=classify_sample, bg="#2980b9", fg="white").pack(pady=5)

    window.mainloop()
