import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=1.0):
    distance = np.sum((x1 - x2) ** 2)
    return np.exp(-gamma * distance)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

def compute_Q(X, y, kernel_function):
    l = X.shape[0]
    Q = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            Q[i, j] = y[i] * y[j] * kernel_function(X[i], X[j])
    return Q

def objective(λ, Q):
    return 0.5 * np.dot(λ, np.dot(Q, λ)) - np.sum(λ)


def decision_function(X_new, X_train, y_train, λ_optimal, kernel_function, b, **kernel_params):
    decision_values = np.zeros(len(X_new))
    for i in range(len(X_new)):
        s = 0
        for alpha, y_i, x_i in zip(λ_optimal, y_train, X_train):
            if alpha > 1e-5:
                s += alpha * y_i * kernel_function(X_new[i], x_i, **kernel_params)
        decision_values[i] = s + b
    return decision_values

def svm_dual_with_visualization(X, y, kernel_function, kernel_params={}, C=1.0):
    l = X.shape[0]
    Q = compute_Q(X, y, kernel_function, **kernel_params)

    def objective(λ, Q):
        return 0.5 * np.dot(λ, np.dot(Q, λ)) - np.sum(λ)

    bounds = [(0, C) for _ in range(len(y))]
    cons = {'type': 'eq', 'fun': lambda λ: np.dot(λ, y)}
    initial_λ = np.zeros(len(y))

    result = minimize(fun=objective,
                      x0=initial_λ,
                      args=(Q,),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=cons)

    λ_optimal = result.x

    support_vector_indices = np.where(λ_optimal > 1e-5)[0]
    support_vectors = X[support_vector_indices]

    b = 0
    for idx in support_vector_indices:
        s = 0
        for alpha, y_i, x_i in zip(λ_optimal, y, X):
            if alpha > 1e-5:
                s += alpha * y_i * kernel_function(X[idx], x_i, **kernel_params)
        b += y[idx] - s
    b /= len(support_vector_indices)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = decision_function(xy, X, y, λ_optimal, kernel_function, b, **kernel_params)
    Z = Z.reshape(XX.shape)

    plt.contourf(XX, YY, Z, levels=[-np.inf, 0, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.2)
    plt.contour(XX, YY, Z, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

    plt.title(f'SVM with {kernel_function.__name__}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    return λ_optimal, kernel_function, b, kernel_params
