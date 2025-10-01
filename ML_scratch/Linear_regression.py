import numpy as np
from numpy.typing import NDArray

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: NDArray = None
        self.bias: float = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # forward
            y_pred = self.predict(X) # [n_samples, n_features] @ [n_features] -> [n_samples]
            
            # compute gradients
            dw = (1/n_samples) * (X.T @ (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # update weights and bias   
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias