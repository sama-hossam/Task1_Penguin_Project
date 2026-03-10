import matplotlib.pyplot as plt
import numpy as np

def decision_boundary(model,X_test,Y_test,feat1,feat2):
     
     
     x1_min=X_test[:,0].min()
     x1_max=X_test[:,0].max()
     x1=np.array([x1_min,x1_max])
     if (model.use_bias):
        x2=-(x1* model.w1+model.w0)/model.w2
     else:
         x2=-(x1* model.w1)/model.w2


     plt.figure(figsize=(8,8))
     plt.scatter(X_test[Y_test==1][:,0],X_test[Y_test==1][:,1],color='red',label='Class 1')  
     plt.scatter(X_test[Y_test==-1][:,0],X_test[Y_test==-1][:,1],color='blue',label='Class 2')
     plt.plot(x1,x2,label='boundary line',color='black')
     plt.xlabel(feat1)
     plt.ylabel(feat2)
     plt.legend()
     plt.grid()
     plt.show()

              
