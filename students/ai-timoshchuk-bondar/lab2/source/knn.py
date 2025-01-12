import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time 

def plot_risk(k_values, loo_errors):
    """ Строит график эмпирического риска в зависимости от k. """
    plt.plot(k_values, loo_errors, marker='o')
    plt.title('Эмпирический риск в зависимости от k')
    plt.xlabel('k')
    plt.ylabel('Ошибка LOO')
    plt.grid(True)
    plt.show()


def dist_matr(X_train:np.array, X_test:np.array, dist_type="d2"):
    if dist_type == "d2":
        squared_norms_obych = np.sum(X_train ** 2, axis=1).reshape(-1, 1)
        squared_norms_predict = np.sum(X_test ** 2, axis=1).reshape(-1, 1)
        dist_matrix = (squared_norms_predict + squared_norms_obych.T) - (2 * (X_test @ X_train.T))
    else:
        raise Exception("Метод не добавлен")
    return dist_matrix


def gaussian_kernel(distance, bandwidth):
    """ Гауссово ядро с переменной шириной (ширина окна — bandwidth). """
    bandwidth = bandwidth.reshape(-1, 1)
    return np.exp(- (distance ** 2) / (2 * bandwidth ** 2))


def find_winning_class(weights, labels, num_classes):
    num_rows = weights.shape[0]


    # Создаем матрицу для накопления весов для каждого класса
    class_weight_sums = np.zeros((num_rows, num_classes))

    # Суммируем веса по классам
    np.add.at(class_weight_sums, (np.arange(num_rows)[:, None], labels), weights)

    # Находим индексы классов с максимальной суммой весов для каждой строки
    winning_classes = np.argmax(class_weight_sums, axis=1)
    
    return winning_classes

def knn_predict(X_train, y_train, X_test, k):
    """ KNN с окном Парзена и переменной шириной окна для одного тестового примера. """
    distances = dist_matr(X_train, X_test)
    sorted_indices_per_point = np.argsort(distances, axis=1)
    k_nearest_indices = sorted_indices_per_point[:, :k+1]

    k_nearest_distances = distances[np.arange(distances.shape[0])[:, None], k_nearest_indices]
    k_nearest_labels = y_train[k_nearest_indices[:, :-1]]
    
    # Используем расстояние до k-го соседа как ширину окна
    bandwidth = k_nearest_distances[:, -1]
    
    # Рассчитываем веса с помощью гауссова ядра
    weights = gaussian_kernel(k_nearest_distances[:, :-1], bandwidth)

    unique_labels = np.unique(y_train)

        # Классификация по взвешенному голосованию
    k_sosed_weight = weights * k_nearest_distances[:, :-1]
    

    return find_winning_class(weights=k_sosed_weight, labels=k_nearest_labels, num_classes=unique_labels.shape[0])


def loo_cross_validation(X, y, k_values):
    """ Подбор параметра k методом скользящего контроля (LOO). """
    n = len(y)
    print(X)
    dimension = X.shape[1]
    loo_errors = np.zeros(len(k_values))
    
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(-1, dimension)
        y_true = y[i]
        
        for j, k in enumerate(k_values):
            # print(X_train.shape, y_train.shape, X_test.shape)
            y_pred = knn_predict(X_train, y_train, X_test, k)
            if y_pred != y_true:
                loo_errors[j] += 1
    
    loo_errors /= n  # Нормализуем на количество примеров
    return loo_errors

# Моя реализация KNN с окном Парзена
def knn_accuracy_custom(X_train, y_train, X_test, y_test, k):
    correct_predictions = 0
    y_pred = knn_predict(X_train, y_train, X_test, k)
    correct_predictions = np.sum(y_pred == y_test)
    return correct_predictions / len(X_test)




if __name__ == "__main__":
    X, y = load_iris(return_X_y=True, as_frame=True)
    X = X.to_numpy()
    feature_indexes = [2, 3]
    X= X[:,feature_indexes]
    y=y.to_numpy()
    print(X.shape)

    
    train_size = 150
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    

    # Подбор оптимального k с использованием LOO
    k_values = range(1, 20)  # Проверим значения k от 1 до 20
    loo_errors = loo_cross_validation(X, y, k_values)

    # Построение графика эмпирического риска
    plot_risk(k_values, loo_errors)

    # Оптимальное значение k
    optimal_k = k_values[np.argmin(loo_errors)]
    print(f'Оптимальное значение k: {optimal_k}')

    start_time = time.time()
    y_pred = knn_predict(X, y, X, optimal_k)
    
    print(f"Время работы нашей реализации: {(time.time()-start_time):.4f} секунд")
    custom_accuracy = knn_accuracy_custom(X, y, X, y, optimal_k)
    print(f"Точность нашей реализации KNN: {custom_accuracy:.4f}")
    
    from sklearn.neighbors import KNeighborsClassifier
    knn_sklearn = KNeighborsClassifier(n_neighbors=optimal_k)

    # Оцениваем точность sklearn KNN с кросс-валидацией
    start_time = time.time()
    knn_sklearn.fit(X, y)
    sklearn_y = knn_sklearn.predict(X)
    print(f"Время работы KNN в sklearn: {(time.time()-start_time):.4f} секунд")
    sklearn_accuracy = knn_sklearn.score(X, y)
    print(f"Точность реализации KNN в sklearn: {sklearn_accuracy:.4f}")



    for group in np.unique(y):
        group = np.array(group)
        plt.scatter(X[y_pred==group, 0], X[y_pred==group, 1] )
    plt.show()

    for group in np.unique(y):
        group = np.array(group)
        plt.scatter(X[sklearn_y==group, 0], X[sklearn_y==group, 1] )
    plt.show()