import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ParzenWindowClassifier:
    def __init__(self, neighbors=1):
        self.training_data = None
        self.training_labels = None
        self.neighbors = neighbors

    def train(self, data, labels):
        self.training_data = np.array(data)
        self.training_labels = np.array(labels)

    def calculate_distance(self, test_point):
        return np.linalg.norm(self.training_data - test_point, axis=1)

    @staticmethod
    def gaussian_kernel(value):
        return np.exp(-0.5 * value**2) / np.sqrt(2 * np.pi)

    def _classify(self, test_instance):
        distances = self.calculate_distance(test_instance)

        sorted_indices = np.argsort(distances)

        window_width = distances[sorted_indices[self.neighbors]] + 1e-4

        class_weights = {}
        for i in range(len(self.training_data)):
            label = self.training_labels[i]
            normalized_distance = distances[i] / window_width
            weight = self.gaussian_kernel(normalized_distance)
            class_weights[label] = class_weights.get(label, 0) + weight

        return max(class_weights, key=class_weights.get)

    def predict(self, test_data):

        test_data = np.array(test_data)
        predictions = [self._classify(instance) for instance in test_data]
        return pd.Series(predictions)

class LeaveOneOutValidator:
    def __init__(self, classifier, max_neighbors):
        self.classifier = classifier
        self.max_neighbors = max_neighbors
        self.error_rates = {}
        self.optimal_k = None

    def validate(self, data, labels):
        data = np.array(data)
        labels = np.array(labels)

        for neighbors_count in range(1, self.max_neighbors + 1):
            errors = 0
            self.classifier.neighbors = neighbors_count

            for index in range(len(data)):
                test_instance = data[index].reshape(1, -1)
                actual_label = labels[index]

                training_data = np.concatenate([data[:index], data[index+1:]])
                training_labels = np.concatenate([labels[:index], labels[index+1:]])

                self.classifier.train(training_data, training_labels)
                predicted_label = self.classifier.predict(test_instance)

                if predicted_label.iloc[0] != actual_label:
                    errors += 1

            self.error_rates[neighbors_count] = errors / len(data)

        minimal_error = min(self.error_rates.values())
        for neighbors_count, error_rate in self.error_rates.items():
            if error_rate == minimal_error:
                self.optimal_k = neighbors_count
                break

def run_experiment():
    iris_data = pd.read_csv("iris.csv")
    features = iris_data.drop(['species'], axis=1)
    labels = iris_data['species']

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.4, random_state=24
    )

    start_time_custom = datetime.datetime.now()
    knn_model = ParzenWindowClassifier(neighbors=1)
    validator = LeaveOneOutValidator(classifier=knn_model, max_neighbors=10)
    validator.validate(train_features, train_labels)
    optimal_k = validator.optimal_k

    print(f"Optimal k (custom implementation): {optimal_k}")

    knn_model = ParzenWindowClassifier(neighbors=optimal_k)
    knn_model.train(train_features, train_labels)
    predictions = knn_model.predict(test_features)

    end_time_custom = datetime.datetime.now()
    custom_duration = (end_time_custom - start_time_custom).microseconds
    print(f"Execution time (custom implementation): {custom_duration} microseconds")

    accuracy_custom = accuracy_score(test_labels, predictions)
    precision_custom = precision_score(test_labels, predictions, average='weighted')
    recall_custom = recall_score(test_labels, predictions, average='weighted')
    f1_custom = f1_score(test_labels, predictions, average='weighted')

    print(f"Accuracy (custom implementation): {accuracy_custom * 100:.2f}%")
    print(f"Precision: {precision_custom:.4f}")
    print(f"Recall: {recall_custom:.4f}")
    print(f"F1-score: {f1_custom:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(validator.error_rates.keys()), list(validator.error_rates.values()), marker='o', label="Error Rate")
    plt.title("Empirical Error (LOO) for Different k Values", fontsize=14)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.grid(True)

    min_error_k = validator.optimal_k
    min_error_value = validator.error_rates[min_error_k]
    plt.annotate(f"Min Error ({min_error_k}, {min_error_value:.4f})",
                 xy=(min_error_k, min_error_value),
                 xytext=(min_error_k + 1, min_error_value + 0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.legend()
    plt.show()

    start_time_sklearn = datetime.datetime.now()
    grid_search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 10)}, cv=5)
    grid_search.fit(train_features, train_labels)
    best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
    best_knn.fit(train_features, train_labels)
    sklearn_predictions = best_knn.predict(test_features)

    end_time_sklearn = datetime.datetime.now()
    sklearn_duration = (end_time_sklearn - start_time_sklearn).microseconds
    print(f"Optimal k (library implementation): {grid_search.best_params_['n_neighbors']}")
    print(f"Execution time (library implementation): {sklearn_duration} microseconds")

    accuracy_sklearn = accuracy_score(test_labels, sklearn_predictions)
    precision_sklearn = precision_score(test_labels, sklearn_predictions, average='weighted')
    recall_sklearn = recall_score(test_labels, sklearn_predictions, average='weighted')
    f1_sklearn = f1_score(test_labels, sklearn_predictions, average='weighted')
#
    print(f"Accuracy (library implementation): {accuracy_sklearn * 100:.2f}%")
    print(f"Precision: {precision_sklearn:.4f}")
    print(f"Recall: {recall_sklearn:.4f}")
    print(f"F1-score: {f1_sklearn:.4f}")

if __name__ == "__main__":
    run_experiment()
