import numpy as np
from scipy import optimize


class Kernel:
    @staticmethod
    def linear(x1, x2):
        return np.dot(x1, x2)

    @staticmethod
    def rbf(x1, x2):
        diff = x1 - x2
        return np.exp(-np.dot(diff, diff) * len(x1) / 2)

    @staticmethod
    def poly(x1, x2, d=3):
        return np.pow(Kernel.linear(x1, x2) + 1, d)


class SVM:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support_vectors = None
        self.support_alpha_y = None

    def fit(self, X, y):
        N = len(y)
        # Gram matrix of h(x) y
        Xy = X * y[:, np.newaxis]
        hXX = np.apply_along_axis(
            lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X,
        )
        yp = y.reshape(-1, 1)
        GramHXy = hXX * np.matmul(yp, yp.T)

        # Lagrange dual problem
        def Ld0(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))

        # Partial derivate of Ld on alpha
        def Ld0dAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))
        constraints = ({'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

        # Maximize by minimizing the opposite
        opt_res = optimize.minimize(fun=lambda a: -Ld0(GramHXy, a),
                                    x0=np.ones(N),
                                    method='SLSQP',
                                    jac=lambda a: -Ld0dAlpha(GramHXy, a),
                                    constraints=constraints)
        self.alpha = opt_res.x
        epsilon = 1e-8
        support_indices = self.alpha > epsilon
        self.support_vectors = X[support_indices]
        self.support_alpha_y = y[support_indices] * self.alpha[support_indices]

        support_labels = y[self.alpha > epsilon]
        self.w = np.sum((self.alpha[:, np.newaxis] * Xy), axis=0)
        self.intercept = support_labels[0] - np.matmul(self.support_vectors[0].T, self.w)

    def predict_one(self, x):
        x1 = np.apply_along_axis(lambda s: self.kernel(s, x), 1, self.support_vectors)
        x2 = x1 * self.support_alpha_y
        return np.sum(x2)

    def predict(self, X):
        d = np.apply_along_axis(self.predict_one, 1, X)
        return 2 * (d > 0) - 1
