import numpy as np
from tqdm import tqdm


class KNNWithParzen:
    def __init__(self, kernel_width: float):
        self.kernel_width = kernel_width

    @staticmethod
    def gaussian_kernel(distance: np.ndarray, kernel_width: float) -> np.ndarray:
        """Гауссово ядро для вычисления весов."""
        coefficient = 1 / np.sqrt(2 * np.pi)
        return coefficient * np.exp(-0.5 * (distance / kernel_width) ** 2)

    def predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
        predictions = []

        for test_point in X_test:
            distances = np.linalg.norm(X_train - test_point, axis=1)

            nearest_neighbors_idx = np.argsort(distances)[:k+1]
            # K ближайших соседей
            nearest_distances = distances[nearest_neighbors_idx[:-1]]
            nearest_labels = y_train[nearest_neighbors_idx[:-1]]

            weights = self.gaussian_kernel(nearest_distances, distances[nearest_neighbors_idx[-1]])

            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight

            predictions.append(max(weighted_votes, key=weighted_votes.get))

        return np.array(predictions)

    def loo_cross_validation(self, X: np.ndarray, y: np.ndarray, max_k: int = 150) -> tuple[int, float, list[float]]:
        """Метод скользящего контроля (LOO) для подбора параметра K."""
        n_samples = len(X)
        best_k, best_accuracy = 1, 0
        history = []

        for k in tqdm(range(1, max_k + 1)):
            correct_predictions = 0

            for i in range(n_samples):
                # Обучающая и тестовая выборки для LOO
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i].reshape(1, -1)
                y_test = y[i]

                y_pred = self.predict(X_train, y_train, X_test, k)
                if y_pred[0] == y_test:
                    correct_predictions += 1

            accuracy = correct_predictions / n_samples
            if accuracy > best_accuracy:
                best_k, best_accuracy = k, accuracy

            history.append(1-accuracy)

        return best_k, best_accuracy, history
