from typing import Protocol
import numpy as np
import scipy.optimize


class SVMKernel(Protocol):
    """Protocol defining the interface for SVM kernel functions"""

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function value for two vectors"""
        ...


class LinearKernel:
    """Linear kernel: K(x,x') = <x,x'>"""

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)

    def __str__(self) -> str:
        return "Linear"


class RBFKernel:
    """RBF (Gaussian) kernel: K(x,x') = exp(-gamma||x-x'||^2)"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def __str__(self) -> str:
        return f"RBF(gamma={self.gamma})"


class PolynomialKernel:
    """Polynomial kernel: K(x,x') = (1 + <x,x'>)^degree"""

    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return (1 + np.dot(x1, x2)) ** self.degree

    def __str__(self) -> str:
        return f"Polynomial(degree={self.degree})"


class SVM:
    def __init__(self, kernel: SVMKernel, C: float = 1.0):
        """
        Initialize SVM classifier

        Args:
            kernel: Kernel function to use
            C: Regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.lambdas: np.ndarray | None = None  # Lagrange multipliers
        self.support_vectors: np.ndarray | None = None
        self.support_vector_labels: np.ndarray | None = None
        self.b: float | None = None  # Intercept term

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix for given data"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # Compute diagonal element
            K[i, i] = self.kernel(X[i], X[i])
            # Compute upper triangle and mirror to lower triangle
            for j in range(i + 1, n_samples):
                K[i, j] = self.kernel(X[i], X[j])
                K[j, i] = K[i, j]  # Kernel is symmetric
        return K

    def _objective(self, lambdas: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the dual objective function to minimize:
        -L(λ) = -Σλᵢ + (1/2)ΣΣλᵢλⱼyᵢyⱼK(xᵢ,xⱼ)
        """
        return -np.sum(lambdas) + 0.5 * np.sum(np.outer(lambdas * y, lambdas * y) * K)

    def _objective_gradient(self, lambdas: np.ndarray, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of the dual objective function"""
        return -np.ones_like(lambdas) + np.sum(lambdas * y * K.T, axis=1) * y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit the SVM classifier

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be [-1, 1]")

        n_samples = X.shape[0]

        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == -1)

        # Validate if the problem is solvable
        if n_positive == 0 or n_negative == 0:
            raise ValueError("Cannot solve SVM with only one class")

        # if self.C * min(n_positive, n_negative) < max(n_positive, n_negative):
        # print("Warning: C might be too small for the class imbalance")

        K = self._compute_kernel_matrix(X)

        # Modified optimization setup with looser tolerances
        solution = scipy.optimize.minimize(
            fun=lambda x: self._objective(x, K, y),
            x0=np.zeros(n_samples),
            method="SLSQP",  # Sequential Least Squares Programming
            jac=lambda x: self._objective_gradient(x, K, y),
            bounds=[(0, self.C) for _ in range(n_samples)],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x, y=y: np.sum(x * y),
                }
            ],
            options={
                'maxiter': 1000,
                'ftol': 1e-6,
                'disp': False
            }
        )

        if not solution.success:
            raise ValueError(f"Optimization did not converge: {solution.message}")

        # Get the optimal Lagrange multipliers
        self.lambdas = solution.x

        # Find support vectors (points with λᵢ > 0)
        sv_threshold = 1e-5  # Numerical tolerance
        sv_idx = self.lambdas > sv_threshold

        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y[sv_idx]
        self.lambdas = self.lambdas[sv_idx]

        if len(self.support_vectors) == 0:
            raise ValueError("No support vectors found. Model may not be useful.")

        # Compute intercept term b
        self.b = self._compute_intercept()

        return self

    def _compute_intercept(self) -> float:
        """
        Compute the intercept term using support vectors
        that lie on the margin (0 < λᵢ < C)
        """

        if len(self.support_vectors) == 0:
            return 0.0

        margin_threshold = 1e-5
        margin_idx = (self.lambdas > margin_threshold) & (
            self.lambdas < self.C - margin_threshold
        )

        if np.any(margin_idx):
            # Use first support vector on margin to compute b
            sv = self.support_vectors[margin_idx][0]
            sv_y = self.support_vector_labels[margin_idx][0]

            # b = yᵢ - Σλⱼyⱼk(xⱼ,xᵢ)
            b = sv_y - np.sum(
                self.lambdas
                * self.support_vector_labels
                * np.array([self.kernel(sv, x) for x in self.support_vectors])
            )
        else:
            # Fallback: use average of predictions
            b = 0
            for i, sv in enumerate(self.support_vectors):
                b += self.support_vector_labels[i] - np.sum(
                    self.lambdas
                    * self.support_vector_labels
                    * np.array([self.kernel(sv, x) for x in self.support_vectors])
                )
            b /= len(self.support_vectors)

        return b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X

        Args:
            X: Data points to classify, shape (n_samples, n_features)

        Returns:
            Predicted class labels (-1 or 1)
        """
        decision_values = (
            np.array(
                [
                    np.sum(
                        self.lambdas
                        * self.support_vector_labels
                        * np.array([self.kernel(x, sv) for sv in self.support_vectors])
                    )
                    for x in X
                ]
            )
            - self.b
        )

        return np.sign(decision_values)

    def get_hyperplane_parameters(self) -> tuple[np.ndarray, float]:
        """
        Get the parameters of the separating hyperplane (w, b)
        Only works for linear kernel.
        
        Returns:
            tuple: (w, b) where w is the normal vector and b is the intercept
        """
        if not isinstance(self.kernel, LinearKernel):
            raise ValueError("Hyperplane parameters can only be computed for linear kernel")
            
        if self.support_vectors is None or self.support_vector_labels is None:
            raise ValueError("Model must be fitted first")
            
        # For linear kernel, w = Σᵢ λᵢyᵢxᵢ
        w = np.sum(self.lambdas[:, np.newaxis] * 
                  self.support_vector_labels[:, np.newaxis] * 
                  self.support_vectors, axis=0)
                  
        return w, self.b
