import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.rng = np.random.default_rng(42)
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None # [1]
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def BCE(self, y_pred, y_true):
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = self.rng.normal(size=(n_features))
        self.b = 0.0

        for _ in range(self.n_iters):
            z = X @ self.w + self.b # [n_samples]
            y_pred = self.sigmoid(z)

            # gradients
            dw = (X.T @ (y_pred - y)) / n_samples # [n_features]
            db = np.sum(y_pred - y) / n_samples # scalar

            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self
    
    def predict(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z) >= 0.5


