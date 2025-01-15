import numpy as np
import matplotlib.pyplot as plt


class RidgeRegression:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_ctrl: np.ndarray, y_ctrl: np.ndarray):
        self.alpha, Q, alphas = self.search_alpha(X, y, X_ctrl, y_ctrl)
        print(f"Optimal alpha: {self.alpha}")
        self.w = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y

        plt.plot(np.log10(alphas), Q)
        plt.xlabel("log10(alpha)")
        plt.ylabel("Q functional")
        plt.title("Q functional")
        plt.grid(True)
        plt.show()

    def predict(self, X):
        return X @ self.w

    def search_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_ctrl: np.ndarray,
        y_ctrl: np.ndarray,
    ) -> float:
        U, S, V_T = np.linalg.svd(X, full_matrices=False)
        Q_functionals = np.zeros(1000)
        alphas = np.logspace(-2, 8, 1000)
        for idx, alpha in enumerate(alphas):
            Q_functional = (
                np.linalg.norm(
                    X_ctrl @ V_T @ np.diag((S) / (alpha + S**2)) @ U.T @ y - y_ctrl
                )
                ** 2
            )

            Q_functionals[idx] = Q_functional

        return alphas[np.argmin(Q_functionals)], Q_functionals, alphas
