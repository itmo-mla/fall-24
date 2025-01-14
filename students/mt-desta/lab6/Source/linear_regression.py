import numpy as np


class LinearRegression:
    """Linear Regression model with optional L2 regularization (Ridge Regression).
    
    The model uses Singular Value Decomposition (SVD) to compute the analytical solution,
    which is numerically stable and efficient for small to medium-sized datasets.

    Args:
        tau (float, optional): Regularization parameter. Defaults to 0.
            - If tau = 0: Standard linear regression
            - If tau > 0: Ridge regression with L2 regularization
    """
    def __init__(self, tau=0):
        self.tau = tau
        self.svd = False

    def _svd(self, X):
        """Compute the Singular Value Decomposition of the feature matrix.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        """
        self.V, self.D, self.Ut = np.linalg.svd(X, full_matrices=False)
        self.svd = True

    def fit(self, X, y, tau=None):
        """Fit the linear regression model using SVD.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target vector of shape (n_samples,)
            tau (float, optional): Regularization parameter. If provided, overrides
                the value set in __init__. Defaults to None.
        """
        if tau is not None:
            self.tau = tau
        if not self.svd:
            self._svd(X)

        if self.tau == 0:
            pseudo_inverse = self.Ut.T @ np.linalg.inv(np.diag(self.D)) @ self.V.T
            self.best_weight = pseudo_inverse @ y
        else:
            self.best_weight = self.Ut.T @ \
                np.linalg.inv(np.diag(self.D ** 2) + self.tau * np.eye(self.D.shape[0])) @ \
                np.diag(self.D) @ self.V.T @ y

    def predict(self, X):
        """Make predictions for the input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples,)
        """
        return X @ self.best_weight
    