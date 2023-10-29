import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None  # [m1, m2, m3, ..., mn]
        self.bias = 0  # b in (mx + b)

    def fit(self, X, y):
        self.number_of_samples, self.number_of_features = X.shape
        self.weights = np.zeros(self.number_of_features)
        for i in range(self.iterations):
            y_pred = self.predict(
                X
            )  # for each record yi in y_pred : yi = m1xi1 + m2xi2 + ... + b

            dw = (2 / self.number_of_samples) * (np.dot(X.T, y_pred - y))
            db = (2 / self.number_of_samples) * (np.sum(y - y_pred))

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
