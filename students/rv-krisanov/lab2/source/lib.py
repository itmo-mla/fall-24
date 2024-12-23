import numpy as np


class KNNClassifier:
    x_train: np.ndarray
    y_train: np.ndarray

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

    def predict_bulk(
            self,
            x_test: np.ndarray,
            y_test: np.ndarray,
            class_count: int,
            k: int
    ) -> np.ndarray:
        confusion_matrix = np.zeros(shape=(class_count, class_count), dtype=np.uint)
        for idx, (x_predictable, y_predictable) in enumerate(zip(x_test, y_test)):
            distances = np.linalg.norm(x_predictable[np.newaxis, :] - self.x_train, axis=1)
            nearest_indices = np.argsort(distances)[:k + 1]
            nearest_distances = distances[nearest_indices]
            nuclear_distances = np.exp(- (1 / 2) * nearest_distances[:k] / nearest_distances[k])
            nearest_classes = self.y_train[nearest_indices[:k]]
            class_powers = np.bincount(nearest_classes, weights=nuclear_distances)
            nearest_class = np.argmax(class_powers)
            confusion_matrix[y_predictable, nearest_class] += 1
        return confusion_matrix

    def predict(
            self,
            x_predictable: np.array,
            k: int,
            x_train: np.ndarray = None,
            y_train: np.ndarray = None,
    ):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train
        distances = np.linalg.norm(x_predictable[np.newaxis, :] - x_train, axis=1)
        nearest_indices = np.argsort(distances)[:k + 1]
        nearest_distances = distances[nearest_indices]
        nuclear_distances = np.exp(- (1 / 2) * nearest_distances[:k] / nearest_distances[k])
        nearest_classes = y_train[nearest_indices[:k]]
        class_powers = np.bincount(nearest_classes, weights=nuclear_distances)
        return np.argmax(class_powers)


def leave_one_out(X: np.ndarray, Y: np.ndarray, k_max: int):
    n = len(X)
    start_k = 3
    biased = np.zeros(shape=(k_max,))
    biased[:start_k] = np.nan
    unbiased = np.zeros(shape=(k_max,))
    unbiased[:start_k] = np.nan
    for k in range(start_k, k_max):
        for idx in np.arange(X.shape[0]):
            biased_predicted_y = KNNClassifier.predict(
                x_train=np.delete(X, idx, axis=0),
                y_train=np.delete(Y, idx, axis=0),
                x_predictable=X[idx],
                k=k
            )
            if biased_predicted_y != Y[idx]:
                biased[k] += 1
            unbiased_predicted_y = KNNClassifier.predict(
                x_train=X,
                y_train=Y,
                x_predictable=X[idx],
                k=k
            )
            if unbiased_predicted_y != Y[idx]:
                unbiased[k] += 1
    return biased / n, unbiased / n
