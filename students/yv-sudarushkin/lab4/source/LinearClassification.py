import numpy as np


class LinearClassifier:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = None
        self.bias = 0
        self.velocity_weights = np.zeros(input_dim)
        self.velocity_bias = 0
        self.loss_history = []
        self.recurrent_loss_history = []

    def initialize_weights_random(self):
        self.weights = np.random.randn(self.input_dim)
        self.bias = np.random.randn()

    def initialize_weights_multistart(self, num_starts=10):
        random_weights = np.random.randn(num_starts, self.input_dim)
        random_biases = np.random.randn(num_starts)
        scores = -np.linalg.norm(random_weights, axis=1)

        best_index = np.argmax(scores)
        self.weights = random_weights[best_index]
        self.bias = random_biases[best_index]

    def initialize_weights_norm(self):
        limit = np.sqrt(6.0 / (self.input_dim + 1))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(self.input_dim, 1))

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.sign(self.forward(X))

    def compute_margin(self, X, y):
        scores = self.forward(X)
        return y * scores

    def compute_loss(self, X, y, l2_lambda=0.01):
        margins = self.compute_margin(X, y)
        hinge_loss = np.mean(np.maximum(0, 1 - margins))
        l2_loss = l2_lambda * np.sum(self.weights ** 2)
        total_loss = hinge_loss + l2_loss
        return total_loss

    def recurent_loss(self, Q_old, loss, lambd=0.001):
        return lambd * loss + (1 - lambd) * Q_old

    def compute_gradients(self, X, y, l2_lambda=0.01):
        n_samples = X.shape[0]
        margins = self.compute_margin(X, y)

        indicator = (margins < 1).astype(float)
        dW = -np.dot((indicator * y).T, X) / n_samples + 2 * l2_lambda * self.weights
        db = -np.sum(indicator * y) / n_samples

        return dW, db

    def update_with_momentum(self, dW, db, learning_rate=0.01, momentum=0.9):
        self.velocity_weights = momentum * self.velocity_weights - learning_rate * dW
        self.velocity_bias = momentum * self.velocity_bias - learning_rate * db

        self.weights += self.velocity_weights
        self.bias += self.velocity_bias

    def stochastic_gradient_descent(self, X, y, epochs=100, learning_rate=0.01, momentum=0.9, l2_lambda=0.01):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                xi = X[i:i + 1]
                yi = y[i:i + 1]
                dW, db = self.compute_gradients(xi, yi, l2_lambda)
                self.update_with_momentum(dW, db, learning_rate, momentum)

            current_loss = self.compute_loss(X, y, l2_lambda)
            q_loss = self.recurent_loss(self.recurrent_loss_history[-1] if len(self.recurrent_loss_history) != 0 else 0, current_loss)
            self.loss_history.append(current_loss)
            self.recurrent_loss_history.append(q_loss)
