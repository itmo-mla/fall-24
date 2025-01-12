import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time


def vis_k_err(k_candidates, errors):
    plt.figure(figsize=(8, 5))
    plt.plot(k_candidates, errors, 'bo-', linewidth=2, markersize=5)
    plt.title('rr k')
    plt.xlabel('k')
    plt.ylabel('Ошибка LOO')
    plt.xticks(k_candidates)
    plt.grid(True)
    plt.show()


def calculate_distance_matrix(train_data, test_data, method="euclidean"):
    if method == "euclidean":
        train_sq = np.sum(train_data ** 2, axis=1).reshape(1, -1)
        test_sq = np.sum(test_data ** 2, axis=1).reshape(-1, 1)
        cross_term = np.dot(test_data, train_data.T)
        distances = np.sqrt(test_sq + train_sq - 2 * cross_term)
    else:
        raise ValueError("Haven't other dists")
    return distances


def apply_gaussian_kernel(distances, bandwidths):
    return np.exp(- (distances ** 2) / (2 * (bandwidths ** 2)))


def determine_class(weights, labels, num_classes):
    aggregated_weights = np.zeros((weights.shape[0], num_classes))
    for cls in range(num_classes):
        aggregated_weights[:, cls] = np.sum(weights * (labels == cls), axis=1)
    predicted = np.argmax(aggregated_weights, axis=1)
    return predicted


def knn_classifier(train_X, train_y, test_X, k):
    distances = calculate_distance_matrix(train_X, test_X)
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
    nearest_labels = train_y[nearest_indices]

    bandwidth = nearest_distances[:, -1].reshape(-1, 1)

    weights = apply_gaussian_kernel(nearest_distances, bandwidth)

    unique_classes = np.unique(train_y)
    num_classes = unique_classes.size

    predictions = determine_class(weights, nearest_labels, num_classes)
    return predictions


def leave_one_out_cv(data_X, data_y, k_range):
    n_samples = data_X.shape[0]
    errors = np.zeros(len(k_range))
    num_classes = len(np.unique(data_y))

    for i in range(n_samples):
        X_train = np.delete(data_X, i, axis=0)
        y_train = np.delete(data_y, i)
        X_test = data_X[i].reshape(1, -1)
        y_true = data_y[i]

        for idx, k in enumerate(k_range):
            y_pred = knn_classifier(X_train, y_train, X_test, k)
            if y_pred[0] != y_true:
                errors[idx] += 1

    errors /= n_samples
    return errors


def calculate_accuracy(true_labels, predicted_labels):
    return np.mean(true_labels == predicted_labels)


def plot_class_predictions(data_X, true_y, predicted_y, title="Классификация KNN"):
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(true_y)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for cls in unique_classes:
        plt.scatter(
            data_X[predicted_y == cls, 0],
            data_X[predicted_y == cls, 1],
            c=colors[cls % len(colors)],
            label=f'Класс {cls}'
        )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def custom_knn_accuracy(train_X, train_y, test_X, test_y, k):
    predictions = knn_classifier(train_X, train_y, test_X, k)
    accuracy = calculate_accuracy(test_y, predictions)
    return accuracy


def main():
    # Загрузка набора данных Iris
    iris = load_iris()
    X = iris.data[:, [2, 3]]  # Используем только два признака для визуализации
    y = iris.target
    print(f"Размер данных: {X.shape}")

    # Параметры перекрестной проверки
    k_values = range(1, 21)

    loo_err = leave_one_out_cv(X, y, k_values)

    vis_k_err(k_values, loo_err)

    optimal_k = k_values[np.argmin(loo_err)]
    print(f'Оптимальное значение k: {optimal_k}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30)
    #
    # if X_test.size == 0:
    #     X_test, y_test = X, y

    print("_"*50)
    print("CUSTOM KNN")
    start_time = time.time()
    y_pred_custom = knn_classifier(X_train, y_train, X_test, optimal_k)
    elapsed_time_custom = time.time() - start_time
    accuracy_custom = calculate_accuracy(y_test, y_pred_custom)
    print(f"Время выполнения: {elapsed_time_custom} секунд")
    print(f"Точность CUSTOM KNN: {accuracy_custom:.4f}")

    plot_class_predictions(X_test, y_test, y_pred_custom, title="CUSTOM KNN")

    # Сравнение с реализацией из sklearn
    from sklearn.neighbors import KNeighborsClassifier

    print("_"*50)
    print("ETALON KNN")
    knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
    start_time = time.time()
    knn_model.fit(X_train, y_train)
    elapsed_time_sklearn = time.time() - start_time
    y_pred_sklearn = knn_model.predict(X_test)
    accuracy_sklearn = knn_model.score(X_test, y_test)
    print(f"Время выполнения sklearn KNN: {elapsed_time_sklearn} секунд")
    print(f"Точность KNN (sklearn): {accuracy_sklearn:.4f}")

    # Визуализация результатов sklearn KNN
    plot_class_predictions(X_test, y_test, y_pred_sklearn, title="KNN (sklearn)")


if __name__ == "__main__":
    main()
