import numpy as np
from typing import Literal


class LinearClassifier:
    def __init__(
        self,
        units_out: int,
        initializer: Literal["random", "glorot", "zeros"] = "random",
        optimizer: Literal["default", "momentum", "nag"] = "default",
        lr_mode: Literal["default", "fast"] = "default",
        regularizer: bool = False,
    ):
        self.units_out = units_out
        self.initializer = initializer
        self.optimizer = optimizer
        self.lr_mode = lr_mode
        self.regularizer = regularizer
        self.history = {"loss": [], "Q": []}
        self.Q = None
        self.w = None
        self.b = None
        self.velocity_w = None
        self.velocity_b = None
        self.rng = np.random.default_rng(seed=42)

    def _init_weights(self, n_features: int, n_samples: int) -> None:
        self.b = np.zeros(shape=(1, self.units_out))
        
        if self.initializer == "random":
            limit = 1.0 / (2 * n_samples)
            self.w = self.rng.uniform(low=-limit, high=limit, size=(n_features, self.units_out))
        elif self.initializer == "glorot":
            limit = np.sqrt(6.0 / (n_features + self.units_out))
            self.w = self.rng.uniform(low=-limit, high=limit, size=(n_features, self.units_out))
        elif self.initializer == "zeros":
            self.w = np.zeros(shape=(n_features, self.units_out)) + 1e-18
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

    def _calc_margin(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        scores = X @ w + b
        margin = y * scores
        return margin

    def _misclassification_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, reg_c: float) -> np.ndarray:
        margin = self._calc_margin(X, y, w, b)
        loss_i = np.maximum(0.0, -margin).ravel()

        if self.regularizer:
            loss_i += reg_c * np.linalg.norm(w) ** 2

        return loss_i

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, reg_c: float) -> tuple[np.ndarray, float]:
        margin = self._calc_margin(X, y, w, b)
        mask = (margin <= 0).ravel()
        dw = -(y[mask] * X[mask]).sum(axis=0).reshape(w.shape)

        if self.regularizer:
            dw += 2.0 * reg_c * w

        db = -(y[mask]).sum()
        return dw, db

    def _update_weights(self, X, y, lr: float, reg_c: float, gamma: float = 0.9) -> None:
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(self.w)
        if self.velocity_b is None:
            self.velocity_b = np.zeros_like(self.b)
        if self.optimizer == "default":
            self.w -= lr * self.dw
            self.b -= lr * self.db
        elif self.optimizer == "momentum":
            self.velocity_w = gamma * self.velocity_w + (1 - gamma) * self.dw
            self.velocity_b = gamma * self.velocity_b + (1 - gamma) * self.db
            self.w -= lr * self.velocity_w
            self.b -= lr * self.velocity_b
        elif self.optimizer == "nag":
            w_ahead = self.w - gamma * lr * self.velocity_w
            b_ahead = self.b - gamma * lr * self.velocity_b
            dw_ahead, db_ahead = self._compute_gradients(X, y, w_ahead, b_ahead, reg_c)
            self.velocity_w = gamma * self.velocity_w + (1 - gamma) * dw_ahead
            self.velocity_b = gamma * self.velocity_b + (1 - gamma) * db_ahead
            self.w -= lr * self.velocity_w
            self.b -= lr * self.velocity_b
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 1e-4, reg_c: float = 1e-5, _lambda: float = 0.01, batch_size: int = 42, gamma: float = 0.9) -> None:
        n_samples, n_features = X.shape
        self._init_weights(n_features, n_samples)

        if self.Q is None:
            indices = self.rng.choice(n_samples, size=batch_size, replace=False)
            x_batch_init = X[indices]
            y_batch_init = y[indices]
            loss_init = self._misclassification_loss(x_batch_init, y_batch_init, self.w, self.b, reg_c)
            self.Q = np.mean(loss_init)

        for epoch in range(epochs):
            indices = self.rng.choice(n_samples, size=batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            self.dw, self.db = self._compute_gradients(X_batch, y_batch, self.w, self.b, reg_c)

            if self.lr_mode == "fast":
                lr = 1.0 / (np.linalg.norm(X_batch) ** 2 + 1e-10)

            loss = self._misclassification_loss(X_batch, y_batch, self.w, self.b, reg_c).sum()
            self.Q = _lambda * loss + (1 - _lambda) * self.Q
            self._update_weights(X_batch, y_batch, lr=lr, reg_c=reg_c, gamma=gamma)

            self.history["loss"].append(loss)
            self.history["Q"].append(self.Q)

            print(f"Epoch {epoch:3d} | loss = {loss:.6f} | Q = {self.Q:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.w + self.b
        return np.sign(scores).ravel()



    
