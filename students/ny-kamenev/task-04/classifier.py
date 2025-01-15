import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self, n_features, learning_rate=0.01, l2=None, momentum=None, forget_temp=0.1, start_weights='random', fast_gd=False, margin_based_sampling=False):
        self.weights = None
        self.n_features = n_features
        self.Q = None
        self.v = 0
        self.Q_list = []
        self.learning_rate = learning_rate
        self.start_weights = start_weights
        self.forget_temp = forget_temp
        self.l2 = l2
        self.momentum = momentum
        self.fast_gd = fast_gd
        self.margin_based_sampling = margin_based_sampling

    def initialize_weights(self, start_weights=None):
        if start_weights is None:
            self.weights = np.random.uniform(-1 / (2 * self.n_features), 1 / (2 * self.n_features), (1, self.n_features))
        else:
            self.weights = start_weights.reshape(1, self.n_features)

    def multistart(self, X, Y, num_iterations):
        attempts = 20
        accuracy = -1
        weights = None
        for _ in range(attempts):
            self.Q_list = []
            self.initialize_weights()
            self.train(X, Y, num_iterations)
            pred = self.predict(X)
            acc = accuracy_score(Y, pred)
            if acc >= accuracy:
                accuracy = acc
                weights = self.weights
        self.initialize_weights(weights)

    def fit(self, X, Y, num_iterations):
        if self.start_weights == 'random':
            self.initialize_weights()
        elif self.start_weights == 'correlation':
            f = np.sum(np.array(X), axis=0)
            self.initialize_weights(np.sum(Y) * f / (f * f))
        elif self.start_weights == 'multistart':
            self.multistart(X, Y, num_iterations)
            return
        else:
            raise ValueError('start_weights can only be "random", "correlation" or "multistart"')

        self.train(X, Y, num_iterations)

    def train(self, X, Y, num_iterations):
        if self.Q is None:
            #Инициализируем оценку функционала на случайных 10 примерах
            if isinstance(Y, pd.Series):
                Y = Y.to_numpy()
            indexes = np.random.choice(range(len(X)), size=(10))
            subX = X[indexes]
            subY = Y[indexes]
            self.Q = np.mean([self.compute_margin_loss(self.weights, x, y) for (x, y) in zip(subX, subY)])


        for _ in range(num_iterations):
            if self.margin_based_sampling:
                if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                    X = X.to_numpy()
                if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
                    Y = Y.to_numpy()

                margins = np.array(self.calculate_margin(X, Y))
                inv_margins = np.max(np.abs(margins)) - np.abs(margins)
                inv_margins = inv_margins / np.sum(np.abs(inv_margins))
                index = np.random.choice(np.arange(len(X)), p=inv_margins)
                x, y = X[index], Y[index]
            else:
                x, y = random.choice(list(zip(X, Y)))

            y = np.array([y]).reshape(-1, 1)
            loss = self.compute_margin_loss(self.weights, x, y)

            self.update_params(x, y)


            self.Q = self.forget_temp * loss + (1 - self.forget_temp) * self.Q
            self.Q_list.append(self.Q[0, 0])

    def update_params(self, x, y):

        if self.fast_gd:
            self.learning_rate = 1 / sum(x ** 2) + 1e-8

        if self.momentum is not None and self.l2 is not None:
            # Нестерова + L2-регуляризация
            regularization_term = self.weights * self.l2
            temp_weights = self.weights - self.learning_rate * self.momentum * self.v
            self.v = (self.momentum * self.v
                      + (1 - self.momentum)
                      * (self.compute_margin_loss_derivative(temp_weights, x, y) + regularization_term))
            self.weights -= self.learning_rate * self.v
        elif self.momentum is not None:
            # Только Нестерова
            temp_weights = self.weights - self.learning_rate * self.momentum * self.v
            self.v = (self.momentum * self.v
                      + (1 - self.momentum)
                      * self.compute_margin_loss_derivative(temp_weights, x, y))
            self.weights -= self.learning_rate * self.v
        elif self.l2 is not None:
            # Только L2-регуляризация
            regularization_term = self.weights * self.l2
            self.weights -= self.learning_rate * (
                    self.compute_margin_loss_derivative(self.weights, x, y) + regularization_term)

        else:
            self.weights -= self.learning_rate * self.compute_margin_loss_derivative(self.weights, x, y)

    def predict(self, X):
        return np.sign(X @ self.weights.T)

    @staticmethod
    def compute_margin_loss(weights, X, y):
        X = X.reshape(1, -1)
        margin = np.dot(weights, X.T) * y
        return (1 - margin) ** 2

    @staticmethod
    def compute_margin_loss_derivative(weights, X, y):
        X = X.reshape(1, -1)
        margin = np.dot(weights, X.T) * y
        return -2 * (1 - margin) * np.dot(y, X)

    def calculate_margin(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        if isinstance(y, pd.Series):
            y = np.array(y)

        self.num_samples, self.num_features = X.shape
        margin_values = []

        for i in range(self.num_samples):
            X_sample = X[i]
            y_sample = y[i]
            X_sample = X_sample.reshape(-1, 1)

            margin_values.append(float(np.dot(X_sample.T, self.weights.T) * y_sample))
        return margin_values

    def plot_results(self, X, y, name, uncertainty_threshold=0.3):
        margins = self.calculate_margin(X, y)
        margins = np.array(margins)
        margins = np.sort(margins.flatten())
        margins = np.clip(margins, -1, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(margins, c='k', linewidth=1)
        ax1.axhline(y=0, c='k', linewidth=0.5)

        x = np.arange(len(margins))

        ax1.fill_between(x, margins, where=(margins >= uncertainty_threshold), color='green', label="Надежные")
        ax1.fill_between(x, margins, where=(margins <= -uncertainty_threshold), color='red', label="Шумовые")
        ax1.fill_between(x, margins,
                         where=np.bitwise_and(margins >= -uncertainty_threshold, margins <= uncertainty_threshold),
                         color='yellow', label="Пограничные")
        ax1.legend()
        ax1.set_title("Margins")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Margin Value")

        ax2.plot(self.Q_list, c='blue', label='Quality Functional (Q)')
        ax2.legend()
        ax2.set_title("Training Progress")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Q Value")

        plt.tight_layout()
        plt.savefig(f"./img/{name}.png")
        plt.close(fig)

    def plot_margin(self, X, y, name, uncertainty_threshold=0.3):
        margins = self.calculate_margin(X, y)

        margins = np.array(margins)
        margins = np.sort(margins.flatten())
        margins = np.clip(margins, -1, 1)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(margins, c='k', linewidth=1)
        ax.axhline(y=0, c='k', linewidth=0.5)

        x = np.arange(len(margins))

        ax.fill_between(x, margins, where=(margins >= uncertainty_threshold), color='green', label="Надежные")
        ax.fill_between(x, margins, where=(margins <= -uncertainty_threshold), color='red', label="Шумовые")
        ax.fill_between(x, margins,
                        where=np.bitwise_and(margins >= -uncertainty_threshold, margins <= uncertainty_threshold),
                        color='yellow', label="Пограничные")
        ax.legend()

        plt.savefig(f"./img/{name}.png")
        plt.close(fig)

    def plot_training_progress(self, name):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(self.Q_list, c='blue', label='Quality Functional (Q)')
        ax.legend()

        plt.savefig(f"./img/{name}.png")
        plt.close(fig)
