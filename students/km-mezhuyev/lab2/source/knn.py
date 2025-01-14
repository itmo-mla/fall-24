import numpy as np
import time

class KNNParzenWindow:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    @staticmethod
    def gaussian_kernel(distances, bandwidth):
        return np.exp(-0.5 * (distances / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, k):
        n_train = self.X_train.shape[0]
        n_test = X_test.shape[0]

        y_pred = np.zeros(n_test)

        for i in range(n_test):
            distances = np.array([self.euclidean_distance(self.X_train[j], X_test[i]) for j in range(n_train)])
            sorted_indices = np.argsort(distances)
            nearest_neighbors = sorted_indices[:k]

            weights = self.gaussian_kernel(distances[nearest_neighbors], self.bandwidth)

            class_weights = {}
            for idx, j in enumerate(nearest_neighbors):
                if self.y_train[j] not in class_weights:
                    class_weights[self.y_train[j]] = 0
                class_weights[self.y_train[j]] += weights[idx]

            y_pred[i] = max(class_weights, key=class_weights.get)

        return y_pred

def loo_cross_validation(X, y, k_values, bandwidths):
    best_k = None
    best_bandwidth = None
    best_error = float('inf')
    results = {}
    times = {}

    for bandwidth in bandwidths:
        errors_for_bandwidth = []
        start_time = time.time()

        for k in k_values:
            error = 0
            for i in range(len(X)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                X_test = X[i:i+1]
                y_test = y[i]

                knn = KNNParzenWindow(bandwidth)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test, k)

                if y_pred != y_test:
                    error += 1

            avg_error = error / len(X)
            errors_for_bandwidth.append(avg_error)
            print(f'Bandwidth: {bandwidth}, k: {k}, Error: {avg_error:.4f}')

            if avg_error < best_error:
                best_error = avg_error
                best_bandwidth = bandwidth
                best_k = k
        times[bandwidth] = time.time() - start_time
        results[bandwidth] = errors_for_bandwidth

    return best_k, best_bandwidth, best_error, results, times

def accuracy(y_true, y_pred):
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    return correct_predictions / total_predictions