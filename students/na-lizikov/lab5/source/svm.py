import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import preprocessing

class SupportVectorMachine:
    def __init__(self, kernel_type='linear', C=1.0, gamma=1.0, degree=3):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel_type = kernel_type
        self.alpha = None
        self.bias = None
        self.support_vectors = None
        self.support_labels = None
        self.weights = None
    
    def _rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    def _polynomial_kernel(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.degree

    def _compute_weights(self):
        return np.sum(self.alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)
    
    def fit(self, data, labels):
        data = np.asarray(data)
        labels = np.where(np.asarray(labels) <= 0, -1, 1)
        n_samples, n_features = data.shape

        def kernel_matrix():
            if self.kernel_type == 'poly':
                return np.array([[self._polynomial_kernel(data[i], data[j]) for j in range(n_samples)] for i in range(n_samples)])
            elif self.kernel_type == 'rbf':
                return np.array([[self._rbf_kernel(data[i], data[j]) for j in range(n_samples)] for i in range(n_samples)])
            else:
                return np.dot(data, data.T)

        def dual_objective(alphas):
            K = kernel_matrix()
            return 0.5 * np.sum((alphas[:, None] * alphas[None, :]) * (labels[:, None] * labels[None, :]) * K) - np.sum(alphas)

        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': lambda alphas: np.dot(alphas, labels)}
        initial_alphas = np.zeros(n_samples)
        solution = minimize(dual_objective, initial_alphas, bounds=bounds, constraints=constraints)
        self.alpha = solution.x
        support_mask = self.alpha > 1e-5
        if not np.any(support_mask):
            raise ValueError("No support vectors found. Check the parameters or the data.")
        self.support_vectors = data[support_mask]
        self.support_labels = labels[support_mask]
        self.alpha = self.alpha[support_mask]
        K = kernel_matrix()
        if len(self.support_labels) > 0:
            self.bias = np.mean(
                self.support_labels - np.sum(self.alpha[:, None] * self.support_labels[:, None] * K[support_mask][:, support_mask], axis=1)
            )
        else:
            self.bias = 0 

    def predict(self, data):
        if self.kernel_type == 'linear':
            self.weights = self._compute_weights()
            return np.sign(np.dot(data, self.weights) + self.bias)
        else:
            results = []
            for x in data:
                result = sum(self.alpha[i] * self.support_labels[i] * (self._polynomial_kernel(x, sv) if self.kernel_type == 'poly' else self._rbf_kernel(x, sv))
                             for i, sv in enumerate(self.support_vectors)) + self.bias
                results.append(np.sign(result))
            return np.array(results)

if __name__ == "__main__":
    data = pd.read_csv("iris.csv")
    label_encoder = preprocessing.LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])
    X = data.drop(['species'], axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernels = ['linear', 'rbf', 'poly']
    results = {}

    for kernel in kernels:
        start_time = time.time()
        svm = SupportVectorMachine(kernel_type=kernel, C=1.0, gamma=0.5, degree=3)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        elapsed_time = time.time() - start_time

        results[kernel] = {'accuracy': accuracy, 'time': elapsed_time}
        print(f"Kernel: {kernel}, Accuracy: {accuracy:.4f}, Time: {elapsed_time:.4f} seconds")

    # Compare with library implementation
    for kernel in kernels:
        start_time = time.time()
        svc = SVC(kernel=kernel, C=1.0, gamma=0.5, degree=3)
        svc.fit(X_train, y_train)
        y_pred_lib = svc.predict(X_test)
        accuracy_lib = accuracy_score(y_test, y_pred_lib)
        elapsed_time_lib = time.time() - start_time

        results[f"{kernel}_lib"] = {'accuracy': accuracy_lib, 'time': elapsed_time_lib}
        print(f"Library Kernel: {kernel}, Accuracy: {accuracy_lib:.4f}, Time: {elapsed_time_lib:.4f} seconds")

    # Visualization
    plt.figure(figsize=(10, 6))
    for i, kernel in enumerate(kernels):
        svc = SVC(kernel=kernel, C=1.0, gamma=0.5, degree=3)
        svc.fit(X_train[:, :2], y_train)  # Use only first two features for visualization

        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(1, 3, i + 1)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(f"Kernel: {kernel}")

    plt.tight_layout()
    plt.show()
