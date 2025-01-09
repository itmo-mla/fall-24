import numpy as np

from scipy.optimize import minimize


def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.dot(x1, x2.T)


def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2.T)) ** degree


def rbf_kernel(x1, x2, gamma=0.1):
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    return np.exp(-gamma * sq_dist)


class SVM:
    def __init__(self, kernel=linear_kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.lambda_values = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
        self.X = None
        self.y = None

    def dual_objective_function(self, lambda_values):
        return -np.sum(lambda_values) + 0.5 * np.sum(
            (self.kernel(self.X, self.X))
            * np.outer(self.y, self.y)
            * np.outer(lambda_values, lambda_values)
        )

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, _n_features = X.shape

        lambda_init = np.random.rand(n_samples)
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x * self.y)}]
        bounds = [(0.0, self.C) for _ in range(n_samples)]

        solution = minimize(
            self.dual_objective_function,
            lambda_init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        self.lambda_values = solution.x

        support_vector_indices = self.lambda_values > 1e-6
        print(f"Найдено опорных векторов: {np.sum(support_vector_indices)}")
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.lambda_values = self.lambda_values[support_vector_indices]

        self.calculate_bias()

        return self

    def calculate_bias(self):
        if len(self.support_vectors) == 0:
            self.b = 0
            return

        return np.mean(
            self.support_vector_labels
            - (
                np.dot(
                    self.lambda_values * self.support_vector_labels,
                    self.kernel(self.support_vectors, self.support_vectors),
                )
            )
        )

    def predict(self, X):
        kernel_matrix = self.kernel(self.support_vectors, X)

        weighted_sum = np.dot(
            self.lambda_values * self.support_vector_labels, kernel_matrix
        )

        return np.sign(weighted_sum - self.b)
