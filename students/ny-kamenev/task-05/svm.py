import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False
class SVM:
    def __init__(self, C, kernel_type, gamma=None, degree=None):
        self.w0 = 0
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree

    def kernel_calc(self, X, X_stroke):
        if self.kernel_type == 'linear':
            return np.dot(X, X_stroke.T)
        if self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.sum((X[:, np.newaxis] - X_stroke) ** 2, axis=2))
        if self.kernel_type == 'polynom':
            return (np.dot(X, X_stroke.T) + 1) ** self.degree


    def fit(self, X_train, y_train):
        kernel = self.kernel_calc(X_train, X_train)
        self.lam = self.optimizer(y_train, self.C, kernel)
        ids = self.lam <= self.C
        self.X_ind = X_train[ids]
        self.Y_ind = y_train[ids]
        self.lam = self.lam[ids]
        kernel_id = kernel[ids][:, ids]
        if self.kernel_type == 'linear':
            self.w = np.sum(self.lam[:, None] * self.Y_ind[:, None] * self.X_ind, axis=0)
            self.w0 = np.mean(self.Y_ind - np.dot(self.X_ind, self.w))
        else:
            self.w0 = np.mean(
                self.Y_ind - np.sum(self.lam[:, None] * self.Y_ind[:, None] * kernel_id, axis=0)
            )
    def predict(self, X):
        kernel = self.kernel_calc(X, self.X_ind)
        if self.kernel_type == 'linear':
            decision = np.dot(X, self.w) + self.w0
        else:
            decision = np.sum(self.lam[:, None] * self.Y_ind[:, None] * kernel.T, axis=0) + self.w0
        return np.sign(decision)

    @staticmethod
    def optimizer(y_train, C, kernel):
        N = len(y_train)
        P = matrix(np.outer(y_train, y_train) * kernel)
        q = matrix(-np.ones(N))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        A = matrix(y_train, (1, N), 'd')
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])


    def plot_predictions(self, X, y, name):
        x0s = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        x1s = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X_pred = np.c_[x0.ravel(), x1.ravel()]
        y_pred = self.predict(X_pred).reshape(x0.shape)

        plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, edgecolor='k')
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(f"./img/{name}.png")
        plt.close()

