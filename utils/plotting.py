import matplotlib.pyplot as plt
import numpy as np

def plot_results(X, y, model, feat1, feat2, title="Decision Boundary"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==1][:,0], X[y==1][:,1], label='Class 1', c='red', marker='o')
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], label='Class 2', c='blue', marker='x')
    
    if model.use_bias:
        x1_min, x1_max = X[:,0].min(), X[:,0].max()
        x1_vals = np.array([x1_min, x1_max])
        x2_vals = -(model.w0 + model.w1 * x1_vals) / model.w2
    else:
        x1_min, x1_max = X[:,0].min(), X[:,0].max()
        x1_vals = np.array([x1_min, x1_max])
        x2_vals = -(model.w1 * x1_vals) / model.w2
    
    plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()