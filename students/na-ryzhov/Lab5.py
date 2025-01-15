import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time

iris = load_iris()
X, y = iris.data, iris.target

class_1, class_2 = 0, 1
mask = (y == class_1) | (y == class_2)
X = X[mask]
y = y[mask]
y = np.where(y == class_1, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

def dual_objective(lambdas, X, y, kernel):
    n = len(y)
    term1 = 0.5 * np.sum([
        lambdas[i] * lambdas[j] * y[i] * y[j] * kernel(X[i], X[j])
        for i in range(n) for j in range(n)
    ])
    term2 = np.sum(lambdas)
    return term1 - term2

def constraint_eq(lambdas, y):
    return np.dot(lambdas, y)

def solve_dual(X, y, kernel, C=None):
    n = len(y)
    bounds = [(0, C) for _ in range(n)] if C else [(0, None) for _ in range(n)]
    initial_lambdas = np.zeros(n)

    constraints = {'type': 'eq', 'fun': constraint_eq, 'args': (y,)}
    result = minimize(
        dual_objective, initial_lambdas,
        args=(X, y, kernel),
        bounds=bounds, constraints=constraints
    )
    return result.x

def compute_weights_and_bias(X, y, lambdas, kernel):
    support_indices = np.where(lambdas > 1e-5)[0]
    weights = np.sum([
        lambdas[i] * y[i] * X[i] for i in support_indices
    ], axis=0)
    bias = np.mean([
        y[i] - np.sum([
            lambdas[j] * y[j] * kernel(X[i], X[j])
            for j in support_indices
        ]) for i in support_indices
    ])
    return weights, bias

def predict(X, weights, bias, kernel=None):
    if kernel:
        return np.sign([
            np.sum([weights[i] * kernel(X[i], x) for i in range(len(weights))]) + bias
            for x in X
        ])
    else:
        return np.sign(np.dot(X, weights) + bias)

linear_kernel = lambda x1, x2: np.dot(x1, x2)
def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def poly_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0)**degree

kernels = {
    "Linear Kernel": linear_kernel,
    "RBF Kernel": lambda x1, x2: rbf_kernel(x1, x2, gamma=0.5),
    "Polynomial Kernel": lambda x1, x2: poly_kernel(x1, x2, degree=3, coef0=1)
}

native_results = {}
sklearn_results = {}

for kernel_name, kernel_func in kernels.items():
    start_time_native = time.time()
    lambdas = solve_dual(X_train_2d, y_train, kernel_func)
    weights, bias = compute_weights_and_bias(X_train_2d, y_train, lambdas, kernel_func)
    predictions = predict(X_test_2d, weights, bias, kernel=kernel_func)
    accuracy = accuracy_score(y_test, predictions)
    native_time = time.time() - start_time_native
    native_results[kernel_name] = (accuracy, native_time)

    start_time_sklearn = time.time()
    svm = SVC(
        kernel='linear' if kernel_name == "Linear Kernel" else
               'rbf' if kernel_name == "RBF Kernel" else
               'poly',
        C=1,
        gamma=0.5 if kernel_name == "RBF Kernel" else 'scale'
    )
    svm.fit(X_train_2d, y_train)
    accuracy_sklearn = svm.score(X_test_2d, y_test)
    sklearn_time = time.time() - start_time_sklearn
    sklearn_results[kernel_name] = (accuracy_sklearn, sklearn_time)

for kernel_name in kernels.keys():
    native_accuracy, native_time = native_results[kernel_name]
    sklearn_accuracy, sklearn_time = sklearn_results[kernel_name]
    print(f"{kernel_name}:")
    print(f"  Native - Accuracy: {native_accuracy:.4f}, Time: {native_time:.4f} seconds")
    print(f"  Sklearn - Accuracy: {sklearn_accuracy:.4f}, Time: {sklearn_time:.4f} seconds")

plt.figure(figsize=(15, 15))
plot_idx = 1

def visualize_decision_boundary(X, y, weights, bias, title, kernel=None):
    global plot_idx
    plt.subplot(3, 2, plot_idx)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], weights, bias, kernel)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.title(title)
    plot_idx += 1

for kernel_name, kernel_func in kernels.items():
    lambdas = solve_dual(X_train_2d, y_train, kernel_func)
    weights, bias = compute_weights_and_bias(X_train_2d, y_train, lambdas, kernel_func)
    visualize_decision_boundary(X_train_2d, y_train, weights, bias, f"Native {kernel_name}", kernel=kernel_func)

for kernel_name in kernels.keys():
    svm = SVC(
        kernel='linear' if kernel_name == "Linear Kernel" else
               'rbf' if kernel_name == "RBF Kernel" else
               'poly',
        C=1e10,
        gamma=0.5 if kernel_name == "RBF Kernel" else 'scale'
    )
    svm.fit(X_train_2d, y_train)
    plt.subplot(3, 2, plot_idx)
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.title(f"Sklearn {kernel_name}")
    plot_idx += 1

plt.tight_layout()
plt.show()
