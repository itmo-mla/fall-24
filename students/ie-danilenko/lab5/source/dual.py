import numpy as np
from scipy.optimize import minimize
from read import read_student
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class SVM:
    def __init__(self, method, C=1.0, gamma=None, r=None, d=None):
        self.method = method
        self.C = C
        self.gamma = gamma
        self.r = r
        self.d = d

    def __linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def __rbf_kernel(self, X1, X2):
        diff = X1[:, np.newaxis] - X2
        sq_dist = np.sum(diff ** 2, axis=2)
        return np.exp(-self.gamma * sq_dist)
    
    def __polynomial_kernel(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + self.r) ** self.d

    def get_k(self, X, y):
        if self.method == 'linear':
            return self.__linear_kernel(X, X)
        elif self.method == 'rbf':
            return self.__rbf_kernel(X, X)
        elif self.method == 'polynomial':
            return self.__polynomial_kernel(X, X)
        return None

    def objective(self, alpha, X, y):
        K = self.get_k(X, y)
        return -np.sum(alpha) + 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K)

    def constraints(self, alpha):
        return np.sum(alpha) - self.C
    
    def get_weight(self):
        return np.sum(self.lambdas[:, np.newaxis] * self.support_vector_labels[:, np.newaxis] * self.support_vectors, axis=0)

    def solve(self, X, y):
        n_samples = X.shape[0]
        initial_alpha = np.zeros(n_samples)
        
        cons = {'type': 'eq', 'fun': self.constraints}
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        result = minimize(self.objective, initial_alpha, args=(X, y), method='BFGS', bounds=bounds, constraints=cons)

        support_vector_mask = result.x > 1e-5
        
        self.support_vectors = X[support_vector_mask]
        self.support_vector_labels = y[support_vector_mask]
        self.lambdas = result.x[support_vector_mask]

        K = self.get_k(self.support_vectors, X[support_vector_mask])
        self.b = np.mean(self.support_vector_labels - np.sum(self.lambdas * self.support_vector_labels * K, axis=0))

    def predict(self, X):
        if self.method == 'linear':
            K = self.__linear_kernel(self.support_vectors, X)
        elif self.method == 'rbf':
            K = self.__rbf_kernel(self.support_vectors, X)
        elif self.method == 'polynomial':
            K = self.__polynomial_kernel(self.support_vectors, X)
        
        return np.sign(np.sum(self.lambdas[:, np.newaxis] * self.support_vector_labels[:, np.newaxis] * K, axis=0) - self.b)


if __name__ == "__main__":
    X, y = read_student('dataset/student.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[0:100]
    y_train = y_train[0:100]
    X_val = X_val[0:100]
    y_val = y_val[0:100]

    svm = SVM('linear', 1.0)
    svm.solve(X_train, y_train)
    predictions = svm.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    print("Result (linear) - Accuracy:", accuracy)

    svm = SVM('rbf', 1.0, 0.5)
    svm.solve(X_train, y_train)
    predictions = svm.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    print("Result (rbf) - Accuracy:", accuracy)

    svm = SVM('polynomial', 1.0, 0.5, 1.0, 2.0)
    svm.solve(X_train, y_train)
    predictions = svm.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    print("Result (polynomial) - Accuracy:", accuracy)
