import numpy as np


def compute_loss_and_gradient(X, w, y, alpha):
    predictions = X @ w
    errors = y - predictions
    loss = 0.5 * np.sum(errors ** 2) + 0.5 * alpha * np.sum(w ** 2)
    grad = -(X.T @ errors) + alpha * w
    return loss, grad


def compute_loss(X, w, y, alpha):
    predictions = X @ w
    errors = y - predictions
    return 0.5 * np.sum(errors ** 2) + 0.5 * alpha * np.sum(w ** 2)


def init_weights_random(n_features):
    return np.random.randn(n_features)


def multi_start_initialization(X, y, alpha, n_starts=5):
    best_w = None
    best_loss = float('inf')
    n_features = X.shape[1]

    for _ in range(n_starts):
        w_candidate = init_weights_random(n_features)
        loss_candidate = compute_loss(X, w_candidate, y, alpha)
        if loss_candidate < best_loss:
            best_loss = loss_candidate
            best_w = w_candidate
    return best_w


def correlation_initialization(X, y):
    w_init = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        w_init.append(corr)
    return np.array(w_init)


class LinearClassifier:
    def __init__(self, alpha=0.1, n_epochs=10, method='fast_gd', initialization='random', n_starts=5, lr=0.01, gamma=0.9):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.method = method
        self.initialization = initialization
        self.n_starts = n_starts

        self.lr = lr
        self.gamma = gamma

        self.w_ = None
        self.losses_ = []


    def compute_margin(self, X, y):
        return y * (X @ self.w_)


    def fit(self, X, y):
        if self.initialization == 'random':
            self.w_ = init_weights_random(X.shape[1])
        elif self.initialization == 'multistart':
            self.w_ = multi_start_initialization(X, y, self.alpha, self.n_starts)
        elif self.initialization == 'correlation':
            self.w_ = correlation_initialization(X, y)
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

        if self.method == 'fast_gd':
            self.fit_fast_gd(X, y)
        elif self.method == 'margin_prob':
            self.fit_margin_prob(X, y)
        elif self.method == 'momentum':
            self.fit_momentum(X, y, lr=self.lr, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self


    def fit_fast_gd(self, X, y):
        n_samples, n_features = X.shape
        w = self.w_.copy()
        self.losses_ = []
        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                pred = w @ x_i
                error = y_i - pred
                grad_i = -x_i * error + self.alpha * w
                h_star = 1.0 / (np.dot(x_i, x_i) + 1e-12)
                w = w - h_star * grad_i
            loss_val = compute_loss(X, w, y, self.alpha)
            self.losses_.append(loss_val)

        self.w_ = w


    def fit_margin_prob(self, X, y):
        lr = 0.01
        n_samples, n_features = X.shape
        w = self.w_.copy()
        self.losses_ = []
        for epoch in range(self.n_epochs):
            margins = self.compute_margin(X, y)
            eps = 1e-12
            abs_margins = np.abs(margins) + eps
            inv_m = 1.0 / abs_margins
            p = inv_m / np.sum(inv_m)
            chosen_indices = np.random.choice(
                np.arange(n_samples),
                size=n_samples,
                replace=True,
                p=p
            )
            for i in chosen_indices:
                x_i = X[i]
                y_i = y[i]
                _, grad_i = compute_loss_and_gradient(x_i.reshape(1, -1), w, np.array([y_i]), self.alpha)
                w = w - lr * grad_i
            loss_val = compute_loss(X, w, y, self.alpha)
            self.losses_.append(loss_val)

        self.w_ = w


    def fit_momentum(self, X, y, lr=0.01, gamma=0.9):
        n_samples, n_features = X.shape
        w = self.w_.copy()
        v = np.zeros_like(w)
        self.losses_ = []
        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                _, grad_i = compute_loss_and_gradient(x_i.reshape(1, -1), w, np.array([y_i]), self.alpha)
                v = gamma * v + lr * grad_i
                w = w - v
            loss_val = compute_loss(X, w, y, self.alpha)
            self.losses_.append(loss_val)
        self.w_ = w


    def predict(self, X):
        return np.sign(X.dot(self.w_))
