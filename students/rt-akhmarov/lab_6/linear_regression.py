import numpy as np

class LinearRegression:
    def __init__(self):
        self.Q = []
        self.tau_range = np.arange(0, 100, 0.001)
        self.best_tau = None

    def _svd(self, X):
        return np.linalg.svd(X, compute_uv=True, full_matrices=False)

    def quality(self, X_train, X_test, y_test):
        y_train = self.predict(X_train)
        return np.linalg.norm(X_test @ self.Vt @ np.diag(self.D) @ self.U.T @ y_train - y_test)**2

    def fit(self, X, y, tau):
        self.U, self.D, self.Vt = self._svd(X)
        self.w = self.Vt.T @ np.linalg.inv(np.diag(self.D**2) + tau * np.eye(len(self.D))) @ np.diag(self.D) @ self.U.T @ y
        return self.w

    def fit_tau(self, X_train, X_test, y_train, y_test):
        self.U, self.D, self.Vt = self._svd(X_train)
        min_Q = np.inf

        for tau in self.tau_range:
            w_tau =  self.Vt.T @ np.diag(self.D)/(tau + self.D**2) @ self.U.T @ y_train
            q = np.sum((X_test @ w_tau - y_test)**2)
            if q < min_Q:
                min_Q = q
                self.best_tau = tau
            self.Q.append(q)

        return self.best_tau
    
    def predict(self, X:np.ndarray, w:np.ndarray=None):
        if w is None:
            return X @ self.w
        else:
            return X @ w