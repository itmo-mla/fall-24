import numpy as np
from pydantic import BaseModel


class TauSelectionResult(BaseModel):
    """Results of tau parameter selection process."""

    optimal_tau: float
    scores: dict[float, float]
    best_score: float


class RidgeRegression:
    """Ridge regression implementation (L2 regularization).

    Minimizes the objective function:
    Q_τ(w) = ||Fw - y||^2 + (τ/2)||w||^2
    """

    def __init__(self, tau: float = 1.0):
        """Initialize Ridge regression model.

        Args:
            tau: Regularization strength (default=1.0). Must be positive.
                Larger values specify stronger regularization.
        """
        if tau < 0:
            raise ValueError("Regularization parameter tau must be positive")
        self.tau = tau
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit Ridge regression model.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: Returns the instance itself
        """
        n_samples, n_features = X.shape

        # Add bias term
        bias_column = np.ones((n_samples, 1))
        features = np.concatenate([bias_column, X], axis=1)

        # Calculate optimal weights using the analytical solution:
        # w* = (F^T F + τI)^(-1) F^T y
        identity = np.eye(n_features + 1)  # +1 for bias term
        self.weights = np.linalg.solve(
            features.T @ features + self.tau * identity, features.T @ y
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Args:
            X: Samples of shape (n_samples, n_features)

        Returns:
            y: Returns predicted values
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before making predictions")

        # Add bias term and calculate predictions
        bias_column = np.ones((X.shape[0], 1))
        features = np.concatenate([bias_column, X], axis=1)

        return features @ self.weights

    def select_tau(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tau_values: np.ndarray,
    ) -> TauSelectionResult:
        """Select optimal regularization parameter using validation set.

        Args:
            X_train: Training features of shape (n_train_samples, n_features)
            y_train: Training targets of shape (n_train_samples,)
            X_val: Validation features of shape (n_val_samples, n_features)
            y_val: Validation targets of shape (n_val_samples,)
            tau_values: Array of regularization parameters to evaluate

        Returns:
            TauSelectionResult containing:
                - optimal_tau: Best regularization parameter
                - scores: Dictionary mapping tau values to their validation scores
                - best_score: Best validation score achieved
        """
        # Add bias term to training and validation features
        F_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
        F_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)

        # Calculate SVD once for training data
        U, S, Vt = np.linalg.svd(F_train, full_matrices=False)

        # Calculate validation scores for each tau
        scores = {}
        best_score = np.inf
        optimal_tau = tau_values[0]  # Initialize to avoid potential reference error

        for tau in tau_values:
            # Calculate weights using SVD components with training data
            S_tau = S / (S**2 + tau)
            w_tau = Vt.T @ (S_tau * (U.T @ y_train))

            # Calculate validation score
            y_pred = F_val @ w_tau
            score = np.sum((y_val - y_pred) ** 2)
            scores[tau] = score
            if score < best_score:
                best_score = score
                optimal_tau = tau

        return TauSelectionResult(
            optimal_tau=optimal_tau, scores=scores, best_score=best_score
        )
