import random

import numpy as np


class QuadraticMarginLoss:
    @staticmethod
    def calculate(w, X, y):
        return (1 - (w @ X) * y) ** 2

    @staticmethod
    def derivative(w, X, y):
        return -2 * (1 - (w @ X) * y) * (y @ X.T)


class LinearClassifier:
    def __init__(self, n_features):
        self.n_features = n_features
        self.quality = None
        self.w = None

        self.v = 0

        self.loss_history = []
        self.quality_history = []

    def init_weights(self, w=None):
        if w is None:
            w = np.random.uniform(
                low=-1 / (2 * self.n_features),
                high=1 / (2 * self.n_features),
                size=(1, self.n_features),
            )
        self.w = w

    def get_sample_for_margins(self, X, Y):
        margins = ((X @ self.w.T) * Y).flatten()
        abs_inv_margins = max(abs(margins)) - abs(margins)
        norm_abs_inv_margins = abs_inv_margins / sum(abs_inv_margins)
        return np.random.choice(np.arange(len(X)), p=norm_abs_inv_margins)

    def fit(self, X, Y, n_iter, lr, lam, reg, momentum, gamma, optimize_lr, use_margins):
        if self.quality is None:
            random_indices = np.random.choice(range(len(X)), size=30)
            self.quality = np.mean([
                QuadraticMarginLoss.calculate(w=self.w, X=x, y=y)
                for (x, y) in zip(X[random_indices], Y[random_indices])
            ])

        for _ in range(n_iter):
            if not use_margins:
                x, y = random.choice(list(zip(X, Y)))
            else:
                indices = self.get_sample_for_margins(X, Y)
                x, y = X[indices], Y[indices]

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            loss = QuadraticMarginLoss.calculate(w=self.w, X=x, y=y)

            if optimize_lr:
                lr = 1 / sum(x ** 2)

            if momentum:
                self.v = (
                    gamma * self.v +
                    (1 - gamma) * QuadraticMarginLoss.derivative(
                        w=self.w - lr * gamma * self.v,
                        X=x,
                        y=y,
                    )
                )
                self.w = (
                    self.w * (1 - lr * reg) -
                    lr * self.v
                )
            else:
                self.w = (
                    self.w * (1 - lr * reg) -
                    lr * QuadraticMarginLoss.derivative(w=self.w, X=x, y=y)
                )

            self.quality = lam * loss + (1 - lam) * self.quality
            self.loss_history.append(loss[0, 0])
            self.quality_history.append(self.quality[0, 0])

    def predict(self, X):
        return np.sign(X @ self.w.T)
