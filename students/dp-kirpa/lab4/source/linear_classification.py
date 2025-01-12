import numpy as np


MAX_VALUE = 500

def clip_gradients(gradient, max_value):
    return np.clip(gradient, -max_value, max_value)

def l2_penalty_gradient(w, alpha=0):
    return alpha * w

def compute_loss_gradient(w, x, y, reg_alpha=0):
    return clip_gradients(2 * (np.dot(w, x) - y) * x + l2_penalty_gradient(w, reg_alpha), MAX_VALUE)

def sgd_momentum(
    X, y, lr=0.01, epochs=50, gamma=0.9, initial_weights=None, reg_alpha=0, selection=None
    ):
    if initial_weights is None:
        w = np.zeros(X.shape[1])
    else:
        w = initial_weights

    v = np.zeros(X.shape[1])

    for epoch in range(epochs):
        if selection:
          X, y = selection(X, y, w)
        # выбрать объект xi из Xl случайным образом
        indices = np.random.permutation(len(y))
        for i in indices:
            gradient = compute_loss_gradient(
                w, X[i], y[i], reg_alpha=reg_alpha
                )
            v = gamma * v + lr * gradient
            w -= v
            # рекуррентный для скорости
            Q = (1 - alpha) * Q + alpha * (np.dot(w, X[i]) - y[i]) ** 2

    return w

def compute_step_length(x, grad, X, y):
    num = np.dot(grad.T, grad)
    den = np.dot((X.dot(grad)).T, X.dot(grad))
    return num / den if den != 0 else 1.0

def steepest_gradient_descent(X, y, epochs, initial_weights=None, selection=None):
    if initial_weights is None:
        w = np.zeros(X.shape[1])
    else:
        w = initial_weights

    for epoch in range(epochs):
        if selection:
          X, y = selection(X, y, w)
        grad = -2 * X.T.dot(y - X.dot(w))
        step_length = compute_step_length(w, grad, X, y)
        w -= step_length * grad

        if np.linalg.norm(step_length * grad) < 1e-6:
            break

    return w


def margin_based_presentation(X, y, w):
    margins = np.array([margin(w, X[i], y[i]) for i in range(len(X))])
    indices = np.argsort(np.abs(margins))
    return X[indices], y[indices]


def correlation_based_initialization(X, y):
    return np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])


def multi_start_training(X, y, method, num_starts=10, epochs=50):
    best_w = None
    best_loss = np.inf

    for _ in range(num_starts):
        initial_w = np.random.randn(X.shape[1])
        w = method(X, y, epochs=epochs, initial_weights=initial_w)
        current_loss = np.mean(np.log(1 + np.exp(-y * (X.dot(w)))))
        if current_loss < best_loss:
            best_loss = current_loss
            best_w = w

    return best_w
