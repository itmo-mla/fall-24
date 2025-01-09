import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, kernel_type='linear', C=1, gamma=0.8, d=3):
        self.kernel_type = kernel_type
        self.C = C
        if kernel_type == "linear":
            self.kernel = lambda x, y: x @ y.T
        elif kernel_type == "rbf":
            self.kernel = lambda x, y: np.exp(-gamma*np.sum((x[:, np.newaxis, :]-y[np.newaxis, ...])**2, axis=2))
        elif kernel_type == "poly":
            self.kernel = lambda x, y: (x @ y.T + 1) ** d
        else: raise ValueError(f"Wrong Kernel Type: {kernel_type}")


    def fit(self, X, y, eps=1e-4):
        N = X.shape[0]

        def Lagrangian(lambdas, G):
            return np.sum(lambdas) - 0.5 * lambdas.T @ G @ lambdas

        def dL_dlambda(lambdas, G):
            return np.ones_like(lambdas) - lambdas @ G
        
        I = np.eye(N)
        constraints = ({'type': 'eq',   'fun': lambda l: l @ y, 'jac': lambda l: y},
                        {'type': 'ineq', 'fun': lambda l: I @ l, 'jac': lambda l: I})

        G = y[:, np.newaxis] * self.kernel(X, X) * y[:, np.newaxis].T

        optRes = minimize(
            fun=lambda l: -Lagrangian(l, G),
            x0=np.zeros(N), 
            method='SLSQP', 
            jac=lambda l: -dL_dlambda(l, G),
            constraints=constraints,
            bounds=[(0, self.C) for _ in range(N)],
            options={"maxiter": 1000}
        )

        print(f"Optimization result: {optRes.message}, status {optRes.status}")
        if not optRes.success: return False

        self.lambdas = optRes.x
        self.Y = y
        self.X = X

        self.idxs = optRes.x > eps
        self.support_vectors = X[self.idxs]

        self.w = np.sum((self.lambdas[:, np.newaxis] * y[:, np.newaxis] * X), axis=0, keepdims=True)
        self.w0 = np.mean(
            np.sum(
                self.lambdas[self.idxs, np.newaxis] * y[self.idxs, np.newaxis] * self.kernel(X[self.idxs], X[self.idxs]), axis=0, keepdims=True
                ) - y[self.idxs, np.newaxis]
        )
        return True


    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            res.append(
                np.sum(self.lambdas[:, np.newaxis] * self.Y[:, np.newaxis] * self.kernel(self.X, X[i].reshape(1, -1)))
            )
        res = np.array(res)
        return 2*(res - self.w0 > 0) - 1 