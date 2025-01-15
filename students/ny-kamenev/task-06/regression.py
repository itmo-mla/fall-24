import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, param=0.001):
        self.param = param
        self.c = None
        self.U = None
        self.D = None
        self.V = None

    def compute_svd(self, X):
        self.U, self.D, self.V = np.linalg.svd(X, full_matrices=False)

    def fit(self, X, y, recompute_svd=False):
        if recompute_svd or self.U is None or self.D is None or self.V is None:
            self.compute_svd(X)

        w = self.D / (self.param + self.D ** 2)
        self.c = self.V.T.dot(np.diag(w)).dot(self.U.T).dot(y)

    def predict(self, X):
        return X.dot(self.c)

    def find_optimal(self, X_train, X_test, y_train, y_test, min=0, max=100, step=0.1, criterion='mse'):
        reg = np.arange(min, max, step)
        q_values = []

        for param in reg:
            self.param = param
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            if criterion == 'mse':
                q_values.append(mean_squared_error(y_test, y_pred))
            elif criterion == 'mae':
                q_values.append(mean_absolute_error(y_test, y_pred))
            else:
                raise ValueError("Unsupported criterion: choose 'mse' or 'mae'")

        plt.figure(figsize=(10, 6))
        plt.plot(reg, q_values, label="Q")
        plt.title("Q vs Regularization Parameter")
        plt.xlabel("Regularization Parameter")
        plt.ylabel(f"Q ({criterion})")
        plt.grid(alpha=0.3)
        plt.savefig("./img/quality.png")

        best_index = np.argmin(q_values)
        self.param = reg[best_index]
        return self.param