import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import precision_score, recall_score, f1_score

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)


def rbf_kernel(X1, X2, gamma=1.0):
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist)


def poly_kernel(X1, X2, degree=3, coef0=1.0, gamma=1.0):
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree


class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3, coef0=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.lmbd_ = None
        self.sv_indices_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.b_ = None
        self.is_linear_ = False

    def _get_kernel_matrix(self, X1, X2):
        if callable(self.kernel):
            return self.kernel(X1, X2)
        else:
            if self.kernel == 'linear':
                self.is_linear_ = True
                return linear_kernel(X1, X2)
            elif self.kernel == 'rbf':
                if self.gamma == 'scale':
                    gamma = 1.0 / (X1.shape[1] * X1.var())
                else:
                    gamma = self.gamma
                return rbf_kernel(X1, X2, gamma=gamma)
            elif self.kernel == 'poly':
                if self.gamma == 'scale':
                    gamma = 1.0 / (X1.shape[1] * X1.var())
                else:
                    gamma = self.gamma
                return poly_kernel(X1, X2, degree=self.degree, coef0=self.coef0, gamma=gamma)
            else:
                raise ValueError("Unknown kernel")

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y.astype(float)
        m = X.shape[0]
        K = self._get_kernel_matrix(X, X)

        def objective(lmbd):
            return 0.5 * np.sum((lmbd * self.y_train_)[:, None] * (lmbd * self.y_train_)[None, :] * K) - np.sum(lmbd)

        def grad(lmbd):
            return np.dot((lmbd * self.y_train_), K * self.y_train_[:, None]) - np.ones(m)
        
        N = X.shape[0]
        I = np.eye(N)

        cons = [
            {'type': 'eq', 'fun': lambda lmbd: np.dot(lmbd, self.y_train_), 'jac': lambda lmbd: self.y_train_},
            {'type': 'ineq', 'fun': lambda lmbd: np.dot(I, lmbd), 'jac': lambda lmbd: I}]
        bounds = [(0, self.C) for _ in range(m)]

        res = minimize(objective,
                       np.zeros(m),
                       jac=grad,
                       constraints=cons,
                       bounds=bounds,
                       options={'disp': False})

        self.lmbd_ = res.x
        self.sv_indices_ = np.where(self.lmbd_ > 1e-6)[0]

        if len(self.sv_indices_) > 0:
            b_vals = []
            for i in self.sv_indices_:
                val = self.y_train_[i] - np.sum(self.lmbd_ * self.y_train_ * K[:, i])
                b_vals.append(val)
            self.b_ = np.mean(b_vals)
        else:
            self.b_ = 0.0

        return self

    def decision_function(self, X):
        K = self._get_kernel_matrix(X, self.X_train_)
        return np.sum((self.lmbd_ * self.y_train_)[None, :] * K, axis=1) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))

data = pd.read_csv('./mushrooms.csv')
y = data['class'].map({'p': -1, 'e': 1}).values
X = data.drop('class', axis=1)

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X = X.values

X = X[:500]
y = y[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import time


start = time.time() 
svm_custom = CustomSVM(C=1.0, kernel='rbf')
svm_custom.fit(X_train, y_train)
elapsed = time.time() - start 
print(f"Затрачено времени: {elapsed} сек")
y_pred_custom = svm_custom.predict(X_test)
acc_custom = np.mean(y_pred_custom == y_test)
print("Custom SVM:", acc_custom)

start = time.time() 
svm_sklearn = SVC(C=1.0, kernel='rbf')
svm_sklearn.fit(X_train, y_train)
elapsed = time.time() - start 
print(f"Затрачено времени: {elapsed} сек")
y_pred_sklearn = svm_sklearn.predict(X_test)
acc_sklearn = np.mean(y_pred_sklearn == y_test)
print("Sklearn SVM:", acc_sklearn)

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
XX, YY = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_2d = np.c_[XX.ravel(), YY.ravel()]

grid_original = pca.inverse_transform(grid_2d)
ZZ_custom = svm_custom.predict(grid_original)
ZZ_custom = ZZ_custom.reshape(XX.shape)


plt.figure(figsize=(6, 5))
plt.contourf(XX, YY, ZZ_custom, alpha=0.2, cmap='bwr')
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='bwr', alpha=0.5)
plt.colorbar(label='Class')
plt.show()

precision_custom = precision_score(y_test, y_pred_custom, average=None, labels=[-1, 1])
recall_custom = recall_score(y_test, y_pred_custom, average=None, labels=[-1, 1])
f1_custom = f1_score(y_test, y_pred_custom, average=None, labels=[-1, 1])

precision_sklearn = precision_score(y_test, y_pred_sklearn, average=None, labels=[-1, 1])
recall_sklearn = recall_score(y_test, y_pred_sklearn, average=None, labels=[-1, 1])
f1_sklearn = f1_score(y_test, y_pred_sklearn, average=None, labels=[-1, 1])

metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score'],
    'Custom(-1)': [precision_custom[0], recall_custom[0], f1_custom[0]],
    'Custom(+1)': [precision_custom[1], recall_custom[1], f1_custom[1]],
    'Sklearn(-1)': [precision_sklearn[0], recall_sklearn[0], f1_sklearn[0]],
    'Sklearn(+1)': [precision_sklearn[1], recall_sklearn[1], f1_sklearn[1]],
})

print("\nМетрики для классов -1 и +1:\n")
print(metrics_df.to_string(index=False))