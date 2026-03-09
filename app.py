import tkinter as tk
from tkinter import ttk
import pandas as pd

from models.slp import PerceptronModel
from models.adaline import AdalineModel
from utils.confusion_matrix import confusion_matrix
from utils.plotting import plot_results
from preprocessing import preprocess_data

model = None


def run_gui():

    window = tk.Tk()
    window.title("Penguins Classifier")
    window.geometry("450x650")

    # -------- Feature Selection --------
    tk.Label(window, text="Feature 1").pack()

    feat1_box = ttk.Combobox(window)
    feat1_box['values'] = (
        "CulmenLength",
        "CulmenDepth",
        "FlipperLength",
        "BodyMass",
        "OriginLocation"
    )
    feat1_box.pack()

    tk.Label(window, text="Feature 2").pack()

    feat2_box = ttk.Combobox(window)
    feat2_box['values'] = (
        "CulmenLength",
        "CulmenDepth",
        "FlipperLength",
        "BodyMass",
        "OriginLocation"
    )
    feat2_box.pack()

    # -------- Class Selection --------
    tk.Label(window, text="Class 1").pack()

    class1_box = ttk.Combobox(window)
    class1_box['values'] = ("Adelie", "Gentoo", "Chinstrap")
    class1_box.pack()

    tk.Label(window, text="Class 2").pack()

    class2_box = ttk.Combobox(window)
    class2_box['values'] = ("Adelie", "Gentoo", "Chinstrap")
    class2_box.pack()

    # -------- Learning Rate --------
    tk.Label(window, text="Learning Rate").pack()

    lr_entry = tk.Entry(window)
    lr_entry.pack()

    # -------- Epochs --------
    tk.Label(window, text="Epochs").pack()

    epoch_entry = tk.Entry(window)
    epoch_entry.pack()

    # -------- MSE Threshold --------
    tk.Label(window, text="MSE Threshold (Adaline)").pack()

    mse_entry = tk.Entry(window)
    mse_entry.pack()

    # -------- Bias --------
    bias_var = tk.IntVar()

    bias_check = tk.Checkbutton(
        window,
        text="Use Bias",
        variable=bias_var
    )
    bias_check.pack()

    # -------- Algorithm --------
    algo_var = tk.StringVar()
    algo_var.set("Perceptron")

    tk.Label(window, text="Algorithm").pack()

    tk.Radiobutton(
        window,
        text="Perceptron",
        variable=algo_var,
        value="Perceptron"
    ).pack()

    tk.Radiobutton(
        window,
        text="Adaline",
        variable=algo_var,
        value="Adaline"
    ).pack()

    # -------- Results --------
    result_label = tk.Label(window, text="")
    result_label.pack(pady=10)

    # -------- Train Function --------
    def train_model():

        global model

        feat1 = feat1_box.get()
        feat2 = feat2_box.get()

        class1 = class1_box.get()
        class2 = class2_box.get()

        lr = float(lr_entry.get())
        epochs = int(epoch_entry.get())

        bias = True if bias_var.get() == 1 else False

        algorithm = algo_var.get()

        mse_threshold = 0
        if mse_entry.get() != "":
            mse_threshold = float(mse_entry.get())

        # preprocessing
        
        X_train, y_train, X_test, y_test = preprocess_data( "C:/Users/Smaa Hossam/Desktop/NN_task1/Task1_Penguin_Project/data/penguins.csv",feat1, feat2, class1, class2)

        # choose model
        if algorithm == "Perceptron":

            model = PerceptronModel(
                lr,
                epochs,
                bias
            )

        else:

            model = AdalineModel(
                lr,
                epochs,
                mse_threshold,
                bias
            )

        model.train(X_train, y_train)
        plot_results(X_test, y_test, model, feat1, feat2)

        accuracy, predictions = model.test(X_test, y_test)

        TP, TN, FP, FN = confusion_matrix(y_test, predictions)
        
        result = f"""
Accuracy: {accuracy:.2f}

TP = {TP}
TN = {TN}
FP = {FP}
FN = {FN}
"""

        result_label.config(text=result)

    # -------- Train Button --------
    train_button = tk.Button(
        window,
        text="Train Model",
        command=train_model
    )

    train_button.pack(pady=20)

    # -------- Single Sample --------
    tk.Label(window, text="Classify Single Sample").pack()

    sample1 = tk.Entry(window)
    sample1.pack()

    sample2 = tk.Entry(window)
    sample2.pack()

    classify_result = tk.Label(window, text="")
    classify_result.pack()

    def classify_sample():

        if model is None:
            classify_result.config(text="Train model first")
            return
        
        # problem here because of the target encoding for OriginLocation, need to handle it properly in the future 

        x1 = float(sample1.get())
        x2 = float(sample2.get())         



        x0 = 1 if bias_var.get() == 1 else 0

        net = model.w0*x0 + model.w1*x1 + model.w2*x2

        y_pred = 1 if net > 0 else 0

        classify_result.config(text=f"Prediction: {y_pred}")

    classify_button = tk.Button(
        window,
        text="Classify",
        command=classify_sample
    )

    classify_button.pack(pady=10)

    window.mainloop()