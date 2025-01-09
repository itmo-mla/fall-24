import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, kernel, C=1, gamma=1, degree=3):
        self.C = C
        self.alpha = None
        self.b = None
        self.gamma = gamma
        self.degree = degree
        self.support_vectors = None
        self.support_vector_labels = None
        self.kernel = kernel
        self.weights = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1)

        # Определение функции для минимизации
        def objective(alpha):
            if self.kernel == 'polinomial':
                K = np.array([[self.polynomial_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
            elif self.kernel == 'rbf':
                K = np.array([[self.rbf_kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
            else:
                K = np.dot(X, X.T)
            return 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K) - np.sum(alpha)

        # Границы для альфа
        bounds = [(0, self.C) for _ in range(n_samples)]
        # Начальное значение альфа
        initial_alpha = np.zeros(n_samples)

        # Оптимизация двойственной задачи
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
        result = minimize(objective, initial_alpha, bounds=bounds, constraints=constraints)

        self.alpha = result.x

        # Индексы векторов поддержки
        support_indices = self.alpha > 1e-5
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        self.alpha = self.alpha[support_indices]

        # Вычисление b
        self.b = np.mean(self.support_vector_labels - np.dot(X[support_indices], self._get_weights()))

    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def polynomial_kernel(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.degree

    def _get_weights(self):
        return np.sum(self.alpha[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0)

    def predict(self, X):
        self.weights = self._get_weights()
        return np.sign(np.dot(X, self.weights) + self.b)
