import numpy as np


class RidgeRegression:
    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau
        self.weights_ = None
        self.svd_ = None

    def _compute_svd(self, X: np.ndarray) -> None:
        self.svd_ = np.linalg.svd(X, full_matrices=False)

    def fit(self, X: np.ndarray, y: np.ndarray, tau: float | None = None) -> 'RidgeRegression':
        if not self.svd_:
            self._compute_svd(X)
        if tau is not None:
            self.tau = tau

        U, S, Vt = self.svd_
        S_inv = np.diag(S / (S ** 2 + self.tau))
        self.weights_ = Vt.T @ S_inv @ U.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.array:
        return X @ self.weights_

    def quality(self, X: np.ndarray, y: np.ndarray) -> np.floating:
        return np.linalg.norm(X @ self.weights_ - y) ** 2
