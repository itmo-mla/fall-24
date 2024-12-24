import numpy as np
from typing import Literal
from scipy.optimize import minimize, LinearConstraint, Bounds

class DualSVM:
    def __init__(
        self,
        kernel_type: Literal['linear', 'rbf', 'poly'] = 'linear',
        C: float = 1.0,
        gamma: float = 1.0,
        degree: int = 3,
        c_poly: float = 1.0
    ):
        self.kernel_type = kernel_type
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.c_poly = c_poly
        self.lmbda_star = None
        self.b = None
        self.w = None
        self.X_train = None
        self.y_train = None

    def _linear_kernel(self, X1, X2):
        return X1 @ X2.T

    def _rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * dists)

    def _poly_kernel(self, X1, X2):
        return (X1 @ X2.T + self.c_poly) ** self.degree

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape

        if self.kernel_type == 'linear':
            K = self._linear_kernel(X, X)
        elif self.kernel_type == 'rbf':
            K = self._rbf_kernel(X, X)
        else:
            K = self._poly_kernel(X, X)

        def objective_function(lmbda):
            lmbda_y = lmbda * y
            return 0.5 * (lmbda_y @ K @ lmbda_y) - np.sum(lmbda)
        
        def objective_gradient(lmbda):
            lmbda_y = lmbda * y
            return (K @ lmbda_y) * y - 1.0

        lc = LinearConstraint(y.reshape(1, -1), 0, 0)
        bnds = Bounds(np.zeros(n_samples), np.full(n_samples, self.C))
        lmbda_init = np.zeros(n_samples)

        res = minimize(
            fun=objective_function,
            x0=lmbda_init,
            jac=objective_gradient,
            bounds=bnds,
            constraints=[lc],
            method='SLSQP',
            options={'disp': False}
        )

        self.lmbda_star = res.x
        eps = 1e-7
        sv_mask = (self.lmbda_star > eps) & (self.lmbda_star < self.C - eps)
        support_indices = np.where(sv_mask)[0]

        if len(support_indices) == 0:
            support_indices = np.where(self.lmbda_star > eps)[0]

        if len(support_indices) == 0:
            self.b = 0.0
        else:
            b_vals = []
            for k in support_indices:
                val = y[k] - np.sum((self.lmbda_star * y) * K[:, k])
                b_vals.append(val)
            self.b = np.mean(b_vals)

        if self.kernel_type == 'linear':
            self.w = np.sum((self.lmbda_star * y)[:, None] * X, axis=0)
        else:
            self.w = None

    def predict(self, X):
        if self.kernel_type == 'linear' and self.w is not None:
            return np.sign(X @ self.w + self.b)
        else:
            if self.kernel_type == 'rbf':
                K_test = self._rbf_kernel(X, self.X_train)
            elif self.kernel_type == 'poly':
                K_test = self._poly_kernel(X, self.X_train)
            else:
                K_test = self._linear_kernel(X, self.X_train)
            dec_vals = K_test @ (self.lmbda_star * self.y_train) + self.b
            return np.sign(dec_vals)
