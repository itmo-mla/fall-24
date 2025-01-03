import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        S_reg = np.diag(S / (S ** 2 + self.alpha))

        self.coef_ = Vt.T @ S_reg @ U.T @ y

    def predict(self, X):
        return np.array(X) @ self.coef_

    @staticmethod
    def select_optimal_alpha(X, y, alphas):
        best_alpha = None
        best_error = float('inf')

        for alpha in alphas:
            model = RidgeRegression(alpha=alpha)
            model.train(X, y)
            predictions = model.predict(X)
            error = np.mean((y - predictions) ** 2)

            if error < best_error:
                best_error = error
                best_alpha = alpha

        return best_alpha
