import matplotlib.pyplot as plt
import numpy as np

def decision_boundary(model,X_train,Y_train,X_test,Y_test,feat1,feat2):
     
     plt.figure(figsize=(8,8))
     
     plt.scatter(X_train[Y_train == 1][:, 0], X_train[Y_train == 1][:, 1], color='red', label='Train: Class 1', edgecolors='k', s=50)
     plt.scatter(X_train[Y_train == -1][:, 0], X_train[Y_train == -1][:, 1], color='blue', label='Train: Class 2', edgecolors='k', s=50)
     plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1],  color='red', marker='x', label='Test: Class 1', s=80)
     plt.scatter(X_test[Y_test == -1][:, 0], X_test[Y_test == -1][:, 1], color='blue', marker='x', label='Test: Class 2', s=80)
     if model.w2 ==0:
        return
     x1_min=X_test[:,0].min()
     x1_max=X_test[:,0].max()
     x1=np.array([x1_min,x1_max])
     if (model.use_bias):
        x2=-(x1* model.w1+model.w0)/model.w2
     else:
         x2=-(x1* model.w1)/model.w2

     plt.plot(x1,x2,label='boundary line',color='black')
     plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
     plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
     plt.xlabel(feat1)
     plt.ylabel(feat2)
     plt.title(f"Decision Boundary ({feat1} vs {feat2})")
     plt.legend()
     plt.grid()
     plt.show()
     
     

    
     
     

              
