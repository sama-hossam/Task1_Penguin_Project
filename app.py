import tkinter as tk
from tkinter import ttk
import numpy as np


from models.slp import PerceptronModel
from models.adaline import AdalineModel
from utils.confusion_matrix import confusion_matrix
from utils.plotting import decision_boundary
from preprocessing import preprocess_data


model = None
current_scale = None


def run_gui():
    window = tk.Tk()
    window.title("🐧 Penguins Advanced Classifier")
    window.geometry("1000x700")  # عرض أكبر ليكون مريحاً للعين
    window.configure(bg="#f0f2f5") # لون خلفية هادئ

    # إعداد الستايل العام للـ Combobox والأزرار
    style = ttk.Style()
    style.theme_use('clam')
    
    # --- Frame 1: Settings (الشمال) ---
    settings_frame = tk.LabelFrame(window, text=" ⚙️ Model Configurations ", font=("Arial", 12, "bold"), 
                                   padx=20, pady=20, bg="white", fg="#2c3e50")
    settings_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    # Feature Selection
    tk.Label(settings_frame, text="Feature 1:", bg="white").grid(row=0, column=0, sticky="w", pady=5)
    feat1_box = ttk.Combobox(settings_frame, values=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"], width=25)
    feat1_box.grid(row=0, column=1, pady=5)

    tk.Label(settings_frame, text="Feature 2:", bg="white").grid(row=1, column=0, sticky="w", pady=5)
    feat2_box = ttk.Combobox(settings_frame, values=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"], width=25)
    feat2_box.grid(row=1, column=1, pady=5)

    # Class Selection
    tk.Label(settings_frame, text="Class 1:", bg="white").grid(row=2, column=0, sticky="w", pady=5)
    class1_box = ttk.Combobox(settings_frame, values=["Adelie", "Gentoo", "Chinstrap"], width=25)
    class1_box.grid(row=2, column=1, pady=5)

    tk.Label(settings_frame, text="Class 2:", bg="white").grid(row=3, column=0, sticky="w", pady=5)
    class2_box = ttk.Combobox(settings_frame, values=["Adelie", "Gentoo", "Chinstrap"], width=25)
    class2_box.grid(row=3, column=1, pady=5)

    # Parameters
    tk.Label(settings_frame, text="Learning Rate:", bg="white").grid(row=4, column=0, sticky="w", pady=5)
    lr_entry = tk.Entry(settings_frame, width=28)
    lr_entry.insert(0, "0.01")
    lr_entry.grid(row=4, column=1, pady=5)

    tk.Label(settings_frame, text="Epochs:", bg="white").grid(row=5, column=0, sticky="w", pady=5)
    epoch_entry = tk.Entry(settings_frame, width=28)
    epoch_entry.insert(0, "100")
    epoch_entry.grid(row=5, column=1, pady=5)

    tk.Label(settings_frame, text="MSE Threshold:", bg="white").grid(row=6, column=0, sticky="w", pady=5)
    mse_entry = tk.Entry(settings_frame, width=28)
    mse_entry.insert(0, "0.001")
    mse_entry.grid(row=6, column=1, pady=5)

    # Algorithm & Bias
    bias_var = tk.IntVar(value=1)
    tk.Checkbutton(settings_frame, text="Use Bias Weight (w0)", variable=bias_var, bg="white").grid(row=7, column=0, columnspan=2, pady=10)

    algo_var = tk.StringVar(value="Perceptron")
    tk.Label(settings_frame, text="Algorithm:", bg="white", font=("Arial", 10, "bold")).grid(row=8, column=0, sticky="w")
    tk.Radiobutton(settings_frame, text="Perceptron", variable=algo_var, value="Perceptron", bg="white").grid(row=8, column=1, sticky="w")
    tk.Radiobutton(settings_frame, text="Adaline", variable=algo_var, value="Adaline", bg="white").grid(row=9, column=1, sticky="w")

    # --- Frame 2: Results & Predictions (اليمين) ---
    right_container = tk.Frame(window, bg="#f0f2f5")
    right_container.grid(row=0, column=1, sticky="nsew", padx=10)

    # Result Display (Top Right)
    result_frame = tk.LabelFrame(right_container, text=" 📊 Performance Metrics ", font=("Arial", 12, "bold"), 
                                 padx=20, pady=20, bg="white", fg="#2c3e50")
    result_frame.pack(fill="x", pady=10)
    
    result_label = tk.Label(result_frame, text="Train a model to see results...", font=("Courier", 10), justify="left", bg="white")
    result_label.pack()

    # Classification (Bottom Right)
    predict_frame = tk.LabelFrame(right_container, text=" 🔍 Single Sample Prediction ", font=("Arial", 12, "bold"), 
                                  padx=20, pady=20, bg="#e8f4fd", fg="#0056b3")
    predict_frame.pack(fill="x", pady=10)

    tk.Label(predict_frame, text="Feature 1 Value:", bg="#e8f4fd").grid(row=0, column=0, pady=5, sticky="w")
    sample1 = tk.Entry(predict_frame, width=20)
    sample1.grid(row=0, column=1, pady=5)

    tk.Label(predict_frame, text="Feature 2 Value:", bg="#e8f4fd").grid(row=1, column=0, pady=5, sticky="w")
    sample2 = tk.Entry(predict_frame, width=20)
    sample2.grid(row=1, column=1, pady=5)

    classify_result = tk.Label(predict_frame, text="Prediction: --", font=("Arial", 12, "bold"), bg="#e8f4fd", fg="#d35400")
    classify_result.grid(row=3, column=0, columnspan=2, pady=10)

    # --- Functions ---
    def train_model():
        global model, current_scale, location_encoder
        try:
            feat1, feat2 = feat1_box.get(), feat2_box.get()
            class1, class2 = class1_box.get(), class2_box.get()
            lr = float(lr_entry.get())
            epochs = int(epoch_entry.get())
            bias = True if bias_var.get() == 1 else False
            algorithm = algo_var.get()
            mse_threshold = float(mse_entry.get()) if mse_entry.get() != "" else 0

            # Preprocessing
            X_train, y_train, X_test, y_test, current_scale = preprocess_data(
                "D:/3rd year/2nd term/Neural Network/Penguin_Project/data/penguins.csv", feat1, feat2, class1, class2)

            # Model Selection
            if algorithm == "Perceptron":
                model = PerceptronModel(lr, epochs, bias)
            else:
                model = AdalineModel(lr, epochs, bias, mse_threshold)

            model.train(X_train, y_train)
            accuracy, predictions = model.test(X_test, y_test)
            TP, TN, FP, FN = confusion_matrix(y_test, predictions)
            
            result = f"Accuracy: {accuracy:.2f}%\n" + "-"*25 + \
                     f"\nConfusion Matrix:\n       | Pred: 1 | Pred: -1\nAct: 1 | TP: {TP:2d}  | FN: {FN:2d}\nAct:-1 | FP: {FP:2d}  | TN: {TN:2d}"
            
            result_label.config(text=result)
            decision_boundary(model, X_train, y_train, feat1, feat2)
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

    def classify_sample():
        global model, current_scale
        if model is None:
            classify_result.config(text="⚠️ Train model first")
            return
        try:
            v1, v2 = sample1.get().strip(), sample2.get().strip()
            
            # القاموس اليدوي للمكان
            loc_map = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}

            def process_input(val, feature_name):
                if feature_name == "OriginLocation":
                    # لو المستخدم كتب الاسم، بنحوله للرقم المقابل
                    return float(loc_map.get(val, val)) # لو هو رقم أصلاً هيفضل زي ما هو
                return float(val)

            x1_p = process_input(v1, feat1_box.get())
            x2_p = process_input(v2, feat2_box.get())
            
            scaled = current_scale.transform([[x1_p, x2_p]])[0]

            x0 = 1 if bias_var.get() == 1 else 0
            net = (model.w0 * x0) + (model.w1 * scaled[0]) + (model.w2 * scaled[1])
            y_pred = 1 if net >= 0 else -1
            
            res_text = f"Prediction: {y_pred} ({class1_box.get() if y_pred == 1 else class2_box.get()})"
            classify_result.config(text=res_text, fg="#27ae60" if y_pred == 1 else "#2980b9")
        except Exception as e:
            classify_result.config(text=f"❌ Error: {str(e)}")

    # الأزرار في أماكنها الصحيحة
    tk.Button(settings_frame, text="🚀 TRAIN MODEL", command=train_model, bg="#27ae60", fg="white", 
              font=("Arial", 11, "bold"), cursor="hand2", padx=20).grid(row=10, column=0, columnspan=2, pady=15)

    tk.Button(predict_frame, text="🎯 CLASSIFY", command=classify_sample, bg="#2980b9", fg="white", 
              font=("Arial", 10, "bold"), cursor="hand2").grid(row=2, column=0, columnspan=2, pady=5)

    window.mainloop()

