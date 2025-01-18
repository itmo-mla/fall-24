import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn import preprocessing

class KNearestNeighbors:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test):
        x_test = np.atleast_2d(x_test)  # Преобразуем в 2D-массив, если нужно
        # Вычисляем евклидовы расстояния
        distances = np.sqrt(
            np.maximum(
                np.sum(x_test**2, axis=1, keepdims=True) +
                np.sum(self.x_train**2, axis=1) -
                2 * np.dot(x_test, self.x_train.T),
                0
            )
        )

        # Получаем индексы ближайших соседей
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
        max_distances = np.max(nearest_distances, axis=1, keepdims=True)
        max_distances[max_distances == 0] = 1e-10
        # Вычисляем веса с использованием гауссового ядра
        weights = np.exp(-0.5 * (nearest_distances / max_distances) ** 2)

        # Голосуем за классы с учетом весов
        votes = []
        for i in range(len(x_test)):
            labels = self.y_train[nearest_indices[i]]
            unique_classes = np.unique(self.y_train)
            weighted_votes = [
                np.sum(weights[i, labels == cls]) for cls in unique_classes
            ]
            votes.append(unique_classes[np.argmax(weighted_votes)])
        return np.array(votes)


def leave_one_out_cross_validation(X, y, k_values):
    X = np.array(X)
    y = np.array(y)
    errors = []

    for k in k_values:
        error_count = 0
        for i in range(len(X)):
            x_test = X[i:i+1]
            y_test = y[i]
            x_train = np.concatenate((X[:i], X[i+1:]), axis=0)
            y_train = np.concatenate((y[:i], y[i+1:]), axis=0)

            knn = KNearestNeighbors(k, x_train, y_train)
            prediction = knn.predict(x_test)[0]
            if prediction != y_test:
                error_count += 1

        error_rate = error_count / len(X)
        errors.append(error_rate)
    best_k = k_values[np.argmin(errors)]
    return best_k, errors


if __name__ == "__main__":
    data = pd.read_csv("KNNAlgorithmDataset.csv")

    data['diagnosis'] = data['diagnosis'].replace({'B': 0, 'M':1})
    X_train, X_test, y_train, y_test= train_test_split(data[data.columns[2:-1]], data['diagnosis'], test_size=0.2, random_state=42)

    # Подбор параметра k методом LOO
    k_values = range(1, 21)
    best_k, errors = leave_one_out_cross_validation(X_train, y_train, k_values)

    # Вывод лучшего значения k
    print(f"Оптимальное значение k: {best_k}")

    # Построение графика эмпирического риска
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, marker='o', label='Эмпирический риск')
    plt.title("Эмпирический риск для различных значений k")
    plt.xlabel("Число соседей (k)")
    plt.ylabel("Эмпирический риск")
    plt.legend()
    plt.grid()
    plt.show()

    # Сравнение с библиотечной реализацией
    start_custom = time.time()
    custom_knn = KNearestNeighbors(best_k, X_train, y_train)
    custom_predictions = custom_knn.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    custom_time = time.time() - start_custom

    start_sklearn = time.time()
    sklearn_knn = KNeighborsClassifier(n_neighbors=best_k)
    sklearn_knn.fit(X_train, y_train)
    sklearn_predictions = sklearn_knn.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_time = time.time() - start_sklearn

    # Вывод результатов
    print(f"Результаты пользовательской реализации KNN:")
    print(f"Точность: {custom_accuracy:.4f}, Время выполнения: {custom_time:.4f} секунд")

    print(f"Результаты библиотечной реализации KNN (sklearn):")
    print(f"Точность: {sklearn_accuracy:.4f}, Время выполнения: {sklearn_time:.4f} секунд")