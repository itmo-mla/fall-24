import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def linear_kernel(x1, x2):
    return x1 @ x2.T


def rbf_kernel(x1, x2, gamma=0.5):
    diff = x1[:, None, :] - x2[None, :, :]
    d = np.sum(diff ** 2, axis=-1)
    return np.exp(-gamma * d)


def polynomial_kernel(x1, x2, degree=3):
    return (x1 @ x2.T + 1) ** degree


kernel_function = {
            'linear': linear_kernel,
            'rbf': rbf_kernel,
            'poly': polynomial_kernel
        }

class CustomSVM:
    def __init__(self, kernel='linear', C=1.0):
        """
        :param kernel: Тип ядра ('linear', 'rbf', 'poly').
        :param C: Параметр регуляризации (контроль штрафа за ошибки классификации).
        """
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.bias = 0
        self.weights = None

    def _compute_kernel(self, X1, X2):
        self.kernel_function =  kernel_function[self.kernel]
        kernel_matrix = self.kernel_function(X1, X2)
        return kernel_matrix

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = self._compute_kernel(X, X)

        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)

        def constraint(alpha):
            return np.dot(alpha, y)

        bounds = [(0, self.C) for _ in range(n_samples)]  # Ограничения на альфа: 0 <= \alpha_i <= C
        constraints = {'type': 'eq', 'fun': constraint}  # Линейное равенство

        # Решение задачи оптимизации
        result = minimize(fun=objective, x0=np.zeros(n_samples), bounds=bounds, constraints=constraints)
        self.alpha = result.x

        # Определение опорных векторов
        support_indices = self.alpha > 1e-5
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]
        self.alpha = self.alpha[support_indices]

        self.bias = np.mean(
            self.support_labels - np.sum(self.alpha * self.support_labels * K[support_indices][:, support_indices], axis=1)
        )

        if self.kernel == 'linear':
            self.weights = np.sum(self.alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)

    def predict(self, X):
        """
        :return: Вектор предсказанных меток (-1, 1).
        """
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.weights) + self.bias)
        else:
            K = self._compute_kernel(X, self.support_vectors)
            return np.sign(np.sum(self.alpha * self.support_labels * K, axis=1) + self.bias)
