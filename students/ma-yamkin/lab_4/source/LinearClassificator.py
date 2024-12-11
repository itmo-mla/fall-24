import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd


class LinearClassifier:
    def __init__(self):
        self.weights = None
        self.v = None
        self.num_samples = None
        self.num_features = None
        self.loss = None

    def __call__(self, lr, a, lambda_reg, delta, optimizer, t, w, M=False):
        self.learning_rate = lr
        self.delta = delta
        self.momentum = a
        self.lambda_reg = lambda_reg
        self.optimizer = optimizer
        self.t = t
        self.M = M
        self.w = w

    def init_weights(self, X, y):
        if self.w == 'corr':
            X = np.array(X)
            f = np.sum(X, axis=0)
            self.weights = np.sum(y) * f / (f * f)
            self.v = np.zeros(self.num_features) * 0.01
        else:
            self.weights = np.random.rand(self.num_features) * 0.01
            self.v = np.zeros(self.num_features) * 0.01

    def fit(self, X, y):
        if self.w != 'multi':
            self.feedforward(X, y)
        else:
            loss = []
            w = []

            for i in range(5):
                self.feedforward(X, y)
                loss.append(np.mean(self.loss))
                w.append(self.weights)

            min_loss = 1
            for _ in loss:
                if _ < min_loss:
                    min_loss = _

            self.weights = w[loss.index(min_loss)]

    def feedforward(self, X, y):
        self.num_samples, self.num_features = X.shape
        self.init_weights(X, y)

        X = np.array(X)
        y_ = np.where(np.array(y) <= 0, -1, 1)

        q = [self.count_q(X, y_), self.count_q(X, y_)]

        median = None
        q1 = None
        q3 = None
        if self.M is True:
            g = self.margin(X, y_)
            g = [abs(_) for _ in g]
            median = g.index(float(np.median(g)))

            elem = g[0]
            delta = 0.000001

            num = float(np.quantile(g, 0.25))
            for _ in g:
                if _ - num < delta:
                    elem = _
            q1 = g.index(elem)
            num = float(np.quantile(g, 0.75))
            for _ in g:
                if _ - num < delta:
                    elem = _
            q3 = g.index(elem)

        self.loss = []

        while abs(q[-1] - q[-2]) > self.delta:
            if self.M is False:
                i = randint(0, self.num_samples - 1)
            else:
                i = randint(0, self.num_samples - 1)
                if i < q1:
                    i = randint(0, q1)
                elif q1 < i < median:
                    i = randint(q1, median)
                elif median > i > q3:
                    i = randint(median, q3)
                else:
                    i = randint(median, self.num_samples - 1)

            X_butch = X[i]
            y_butch = y_[i]

            M = self.count_margin(X_butch, y_butch)
            self.loss.append((1 - M) ** 2)

            dloss = self.dcount_margin(X_butch, y_butch)
            grad_w = self.backward(X_butch, y_butch, dloss)
            self.update_params(grad_w, X_butch)

            q.append(self.lambda_reg * M + (1 - self.lambda_reg) * q[-1])

    def count_margin(self, X, y):
        return np.dot(self.weights, X.T) * y

    def dcount_margin(self, X, y):
        return - 2 * (1 - self.count_margin(X, y))

    @staticmethod
    def backward(X, y, dloss):
        grad_w = np.dot(X, dloss) * y
        return grad_w

    def update_params(self, grad_w, X):
        if self.optimizer == 'momentum':
            self.v = self.momentum * self.v - self.learning_rate * grad_w
            self.weights += self.v
        elif self.optimizer == 'l2':
            self.weights -= (grad_w + self.weights * self.t) * self.learning_rate
        elif self.optimizer == 'fast':
            self.learning_rate = 1.0 / (np.sum(X * X) + 1e-8)
            self.weights -= grad_w * self.learning_rate
        else:
            self.weights -= grad_w * self.learning_rate

    def predict(self, X):
        X = np.array(X)
        pred = []
        for i in range(len(X)):
            linear_output = np.dot(X[i], self.weights) - 0.1
            pred.append(int(self.sign(linear_output)))
        return pd.Series(np.where(np.array(pred) >= 0, 1, 0))

    @staticmethod
    def sign(x):
        return np.where(x >= 0, 1, -1)

    def count_q(self, X, y):
        i = randint(0, self.num_samples-1)
        return self.count_margin(X[i], y[i])

    def margin(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
            y = np.where(np.array(y) <= 0, -1, 1)

        self.num_samples, self.num_features = X.shape
        g = []
        self.init_weights(X, y)

        for i in range(self.num_samples):
            X_butch = X[i]
            y_butch = y[i]

            g.append(float((np.dot(X_butch, self.weights)) * y_butch))
        return g

    @staticmethod
    def visualize(X, y, g):
        X = np.array(X)
        y = np.where(np.array(y) <= 0, -1, 1)

        # Визуализация данных
        plt.figure(figsize=(10, 6))

        # Отображение точек данных
        for i in range(len(y)):
            if y[i] == 1:
                plt.scatter(X[i][0], X[i][1], color='blue', marker='o',
                            label='Class 1' if 'Class 1' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(X[i][0], X[i][1], color='red', marker='x',
                            label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Визуализация границы принятия решения
        x_values = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, X.shape[0])
        g.sort()
        y_values = g
        plt.plot(x_values, y_values, color='green', label='Decision boundary')

        plt.title('Отступ')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()
