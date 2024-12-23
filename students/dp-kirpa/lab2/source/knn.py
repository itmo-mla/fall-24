from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.model_selection import LeaveOneOut


### Ширина окна Парзена = расстояние до самого дальнего среди k соседей
def parzen_window_knn(X_train, y_train, X_test, k=5) -> np.array:
    predictions = []

    # считаем попарные расстояния между объектами
    # в виде матрицы len(X_train) x len(X_test)
    distances = pairwise_distances(X_train, X_test)

    # делаем полный проход по тестовым данным
    for i in range(len(X_test)):
        # индексы элементов тренировочной выборки, посортированные
        # по возрастанию расстояния до тестового объекта
        idx_sorted = np.argsort(distances[:, i])
        # ближайшие k соседей из тренировочной выборки
        neighbors_idx = idx_sorted[:k]

        # ширина Парзеновского окна - расстояние до самого дальнего
        # среди k соседей
        h = distances[idx_sorted[k-1], i]

        weights = np.exp(- (distances[neighbors_idx, i] ** 2) / (2 * h ** 2 + 1e-10))

        class_weights = np.zeros(np.max(y_train) + 1)

        for neighbor_idx, weight in zip(neighbors_idx, weights):
            class_weights[y_train.iloc[neighbor_idx]] += weight

        predicted_label = np.argmax(class_weights)
        predictions.append(predicted_label)

    return np.array(predictions)


### Ширина окна Парзена задаётся, как гиперпараметр
def gaussian_kernel(distance, bandwidth=1.0):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def parzen_window_knn_with_bandwidth(X_train, y_train, X_test, k=5, bandwidth=1.0):
    distances = pairwise_distances(X_train, X_test)
    predictions = []

    for i in range(X_test.shape[0]):
        neighbors_idx = np.argsort(distances[:, i])[:k]
        weights = gaussian_kernel(distances[neighbors_idx, i], bandwidth)

        weight_per_class = {}
        for idx, weight in zip(neighbors_idx, weights):
            label = y_train.iloc[idx]
            if label in weight_per_class:
                weight_per_class[label] += weight
            else:
                weight_per_class[label] = weight

        predicted_label = max(weight_per_class, key=weight_per_class.get)
        predictions.append(predicted_label)

    return np.array(predictions)


### Подбор оптимального k
def find_best_k(knn_algo, X, y, max_k):
    loo = LeaveOneOut()
    best_k = 1
    best_accuracy = 0

    accuracies = []

    for k in range(1, max_k + 1):
        all_scores = []

        for train_index, test_index in loo.split(X):
            _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
            _y_train, _y_test = y.iloc[train_index], y.iloc[test_index]

            predict = knn_algo(_X_train, _y_train, _X_test, k=k)
            score = sum(predict == _y_test.to_numpy()) / len(_y_test)
            all_scores.append(score)

        current_accuracy = np.mean(all_scores)

        accuracies.append(current_accuracy)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = k

        print(f'Средняя точность для k={k}: {current_accuracy}')

    return best_k, accuracies
