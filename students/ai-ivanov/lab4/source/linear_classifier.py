import numpy as np
from typing import Literal
from pydantic import BaseModel


class LinearClassifierConfig(BaseModel):
    learning_rate: float = 0.01
    momentum: float = 0.9
    reg_coefficient: float = 0.01
    max_iterations: int = 1000
    initialization: Literal["correlation", "random"] = "correlation"
    random_init_std: float = 0.01
    tolerance: float = 1e-6
    sample_by_margin: bool = False
    fastest_descent: bool = False
    batch_size: int = 32

    def __str__(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.model_dump().items()])


class LinearClassifier:
    def __init__(self, config: LinearClassifierConfig = LinearClassifierConfig()):
        self.config = config
        self.w: np.ndarray | None = None
        self.h: np.ndarray | None = None
        self.loss_history: list[float] = []
        self.margin_history: list[np.ndarray] = []

    def _initialize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        if self.config.initialization == "correlation":
            return np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(n_features)])
        else:
            return np.random.normal(0, self.config.random_init_std, n_features)

    def _margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y * (X @ self.w)

    def _loss(self, margin: np.ndarray) -> float:
        return np.mean(np.maximum(0, 1 - margin) ** 2)

    def _loss_gradient(
        self, X: np.ndarray, y: np.ndarray, margin: np.ndarray
    ) -> np.ndarray:
        # Consider only objects with margin < 1
        mask = margin < 1
        grad = np.zeros_like(self.w)
        if np.any(mask):
            grad = -2 * X[mask].T @ ((1 - margin[mask]) * y[mask])
        return grad / len(y)

    def _sample_by_margin(
        self, X: np.ndarray, y: np.ndarray, margin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample objects with probability inversely proportional to margin"""
        # Convert inputs to numpy arrays if they're pandas objects
        X = np.asarray(X)
        y = np.asarray(y)
        margin = np.asarray(margin)

        weights = 1.0 / (np.abs(margin) + 1e-8)  # Add epsilon to avoid division by zero
        weights /= weights.sum()
        indices = np.random.choice(len(X), size=self.config.batch_size, p=weights)
        return X[indices], y[indices]

    def _fastest_descent_step(self, X: np.ndarray) -> float:
        """
        Calculate optimal step size for quadratic loss using steepest descent method.
        For quadratic loss, the optimal step is 1/||x||^2

        Parameters:
        X: np.ndarray - input data

        Returns:
        float - optimal step size
        """

        # For batch processing, we take the average of optimal steps
        # Alternatively, we could use the norm of the entire batch
        if len(X.shape) == 2:  # If we have a batch of samples
            norms = np.sum(X * X, axis=1)  # Squared L2 norms for each sample
            step = 1.0 / (
                np.mean(norms) + 1e-8
            )  # Add small constant to prevent division by zero
        else:  # Single sample
            step = 1.0 / (np.sum(X * X) + 1e-8)

        # Clip step size for numerical stability
        return np.clip(step, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate class probabilities using the sigmoid function"""
        scores = X @ self.w
        proba = 1 / (1 + np.exp(-scores))

        # [:, 0] - probability of class -1
        # [:, 1] - probability of class 1
        return np.vstack([1 - proba, proba]).T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearClassifier":
        self.w = self._initialize_weights(X, y)
        self.h = np.zeros_like(self.w)
        self.loss_history = []
        self.margin_history = []

        for _ in range(self.config.max_iterations):
            margin = self._margin(X, y)
            self.margin_history.append(margin.copy())

            if self.config.sample_by_margin:
                X_batch, y_batch = self._sample_by_margin(X, y, margin)
                margin_batch = self._margin(X_batch, y_batch)
                grad = self._loss_gradient(X_batch, y_batch, margin_batch)
                learning_rate = (
                    self._fastest_descent_step(X_batch)
                    if self.config.fastest_descent
                    else self.config.learning_rate
                )
            else:
                grad = self._loss_gradient(X, y, margin)
                learning_rate = (
                    self._fastest_descent_step(X)
                    if self.config.fastest_descent
                    else self.config.learning_rate
                )

            grad += self.config.reg_coefficient * self.w

            self.h = self.config.momentum * self.h - learning_rate * grad
            w_new = self.w + self.h

            if np.all(np.abs(w_new - self.w) < self.config.tolerance):
                break

            self.w = w_new

            loss = self._loss(margin)
            self.loss_history.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)
