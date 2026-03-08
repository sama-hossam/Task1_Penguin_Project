import numpy as np

class AdalineModel:
    def __init__(self, learning_rate=0.01, epochs=100, use_bias=True, mse_threshold=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.mse_threshold = mse_threshold
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0

    def activation(self, v):
        return v

    def train(self, X, y):
        if self.use_bias:
            self.w0 = np.random.rand() * 0.01
            self.w1 = np.random.rand() * 0.01
            self.w2 = np.random.rand() * 0.01
        else:
            self.w0 = 0
            self.w1 = np.random.rand() * 0.01
            self.w2 = np.random.rand() * 0.01
        
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                x0 = 1 if self.use_bias else 0
                x1 = X[i][0]
                x2 = X[i][1]
                
                v = (self.w0 * x0) + (self.w1 * x1) + (self.w2 * x2)
                error = y[i] - v
                total_error += error ** 2
                
                if error != 0:
                    self.w0 += self.lr * error * x0
                    self.w1 += self.lr * error * x1
                    self.w2 += self.lr * error * x2
            
            mse = total_error / len(X)
            if mse <= self.mse_threshold:
                break

    def test(self, X_test, y_test):
        correct = 0
        predictions = []
        for i in range(len(X_test)):
            x0 = 1 if self.use_bias else 0
            x1 = X_test[i][0]
            x2 = X_test[i][1]
            
            v = (self.w0 * x0) + (self.w1 * x1) + (self.w2 * x2)
            y_pred = 1 if v >= 0 else -1
            predictions.append(y_pred)
            if y_pred == y_test[i]:
                correct += 1
        
        accuracy = (correct / len(y_test)) * 100
        return accuracy, predictions