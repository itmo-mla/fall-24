import numpy as np


def quality(w, X, y, tau=0):
    return np.linalg.norm(X @ w - y) ** 2 + tau / 2 * np.linalg.norm(w) ** 2


class LinearRegression:
    def __init__(self, tau=0):
        self.tau = tau
        self.svd = False

    def _svd(self, X):
        self.V, self.D, self.Ut = np.linalg.svd(X, full_matrices=False)
        self.svd = True

    def fit(self, X, y, tau=None):
        if tau is not None:
            self.tau = tau
        if not self.svd:
            self._svd(X)

        if self.tau == 0:
            F_pseudo = self.Ut.T @ np.linalg.inv(np.diag(self.D)) @ self.V.T
            self.best_weight = F_pseudo @ y
        else:
            self.best_weight = self.Ut.T @ \
                np.linalg.inv(np.diag(self.D ** 2) + self.tau * np.eye(self.D.shape[0])) @ \
                np.diag(self.D) @ self.V.T @ y

    def predict(self, X):
        return X @ self.best_weight
    