import numpy as np


class MyRidgeRegression:
    def __init__(self, tau=1) -> None:
        self.weights = None
        self.tau = tau
        self.U, self.S, self.Vt = None, None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Sinv = np.diag(S / (S ** 2 + self.tau))
        self.weights = Vt.T @ Sinv @ U.T @ y

    def predict(self, X: np.ndarray) -> np.array:
        return X @ self.weights

    def quality(self, X: np.ndarray, y: np.ndarray):
        return np.linalg.norm(X @ self.weights - y) ** 2