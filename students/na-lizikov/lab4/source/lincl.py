import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler

class LinCL:
    def __init__(self, regularization, weight_init_type, lambda_, gamma, epochs=100):
        self.weight_init_type = weight_init_type
        self.regularization = regularization
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epochs = epochs
        self.weights = None
        self.velocity = None

    def compute_margin(self, X, y):
        return np.dot(X, self.weights) * y

    def compute_loss(self, X, y, weights=None):
        if weights is None:
            weights = self.weights
        residuals = y - np.dot(X, weights)
        regularization_term = self.regularization * np.sum(weights ** 2) / 2
        return np.mean(residuals ** 2) + regularization_term

    def initialize_weights(self, method, X, y, retries=20):
        if method == "corr":
            weights = np.zeros(X.shape[1])
            for feature_idx in range(X.shape[1]):
                weights[feature_idx] = np.dot(y, X[:, feature_idx]) / np.dot(X[:, feature_idx], X[:, feature_idx])
            
            correlation_matrix = np.corrcoef(X, rowvar=False)
            off_diagonal = correlation_matrix - np.diag(np.diag(correlation_matrix))
            if np.any(off_diagonal > 0.7):
                print('Correlation detected')
            return weights
        elif method == "multi":
            best_loss = float('inf')
            best_weights = None
            for _ in range(retries):
                temp_weights = self.initialize_weights('random', X, y)
                current_loss = self.compute_loss(X, y, temp_weights)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = temp_weights
            return best_weights
        elif method == "random":
            bound = 1 / (2 * X.shape[1])
            return np.random.uniform(-bound, bound, X.shape[1])

    def select_indices(self, X, y):
        N = X.shape[0]
        margins = np.abs(self.compute_margin(X, y))
        probabilities = (1 / (1 + margins)) / np.sum(1 / (1 + margins))
        return np.random.choice(N, 10, replace=False, p=probabilities)

    def compute_gradient(self, X, y, weights):
        grad = -2 * np.dot(X.T, (y - np.dot(X, weights))) / len(y)
        grad += self.regularization * weights
        return grad

    def fit(self, X_train, y_train):
        self.weights = self.initialize_weights(self.weight_init_type, X_train, y_train)
        self.velocity = np.zeros_like(self.weights)

        indices = self.select_indices(X_train, y_train)
        q_loss = np.mean(self.compute_loss(X_train[indices], y_train[indices]))

        for _ in range(self.epochs):
            indices = self.select_indices(X_train, y_train)
            step_size = 1 / np.linalg.norm(X_train[indices]) ** 2
            gradient = self.compute_gradient(X_train[indices], y_train[indices], self.weights - step_size * self.gamma * self.velocity)
            self.velocity = self.gamma * self.velocity + (1 - self.gamma) * gradient
            self.weights = self.weights * (1 - step_size * self.regularization) - step_size * self.velocity

            current_loss = self.compute_loss(X_train[indices], y_train[indices])
            q_loss = self.lambda_ * current_loss + (1 - self.lambda_) * q_loss

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

if __name__ == "__main__":
    data = pd.read_csv("breast_data.csv")
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': -1})

    # Normalize feature values
    features = data.columns.drop('diagnosis')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])
    y = data['diagnosis'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train, X_test, y_train, y_test = map(np.array, [X_train, X_test, y_train, y_test])

    model = LogCL(regularization=0.01, weight_init_type="random", lambda_=0.5, gamma=0.9, epochs=100)
    model.fit(X_train, y_train)
    margins = model.compute_margin(X_train, y_train)
    margins_sorted = np.sort(margins)

    plt.figure(figsize=(10, 6))
    plt.plot(margins_sorted, linestyle='-', label="Margins")
    plt.axhline(0, color='red', linestyle='--', label="Decision boundary (M=0)")

    plt.title("Distribution of Margins")
    plt.xlabel("Object Index (sorted)")
    plt.ylabel("Margin", fontsize=12)

    plt.fill_between(range(len(margins_sorted)), margins_sorted, 0, where=(margins_sorted < 0), 
                    color='red', label="(M < 0)")

    plt.fill_between(range(len(margins_sorted)), margins_sorted, 0, where=(margins_sorted > 0), 
                    color='green', label="(M > 0)")

    plt.legend(fontsize=10)
    plt.grid(alpha=1)
    plt.show()

    def evaluate_model(initialization_method, selection_method):
        model = LogCL(regularization=0.01, weight_init_type=initialization_method, lambda_=0.5, gamma=0.9, epochs=100)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, end_time - start_time

    accuracy_corr, time_corr = evaluate_model("corr", "random")

    accuracy_multi, time_multi = evaluate_model("multi", "random")

    accuracy_random, time_random = evaluate_model("random", "random")

    start_time_lib = time.time()
    library_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
    library_model.fit(X_train, y_train)
    library_predictions = library_model.predict(X_test)
    end_time_lib = time.time()
    library_accuracy = accuracy_score(y_test, library_predictions)
    library_time = end_time_lib - start_time_lib

    print("Comparison of methods:")
    print(f"Correlation Initialization: Accuracy = {accuracy_corr:.4f}, Time = {time_corr:.9f}s")
    print(f"Multi-start Initialization: Accuracy = {accuracy_multi:.4f}, Time = {time_multi:.9f}s")
    print(f"Random Initialization: Accuracy = {accuracy_random:.4f}, Time = {time_random:.9f}s")
    print(f"Library SGDClassifier: Accuracy = {library_accuracy:.4f}, Time = {library_time:.9f}s")
