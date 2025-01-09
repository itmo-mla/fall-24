from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import time


def read_dataset(path):
    data = pd.read_csv(path)
    data = data.dropna()
    encoder = LabelEncoder()
    data['species'] = encoder.fit_transform(data['species'])

    return data.iloc[:, :-1].to_numpy(), data['species']


def my_knn(data_points, labels, target_points, k):
    labels = np.array(labels)
    class_labels = np.unique(labels)
    results = []

    for target_point in target_points:
        dist = np.linalg.norm(data_points - target_point, axis=1)
        sorted_indices = np.argsort(dist)
        sorted_distances = dist[sorted_indices]

        scaling_factor = sorted_distances[k + 1]
        kernel_weights = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (sorted_distances / scaling_factor) ** 2)

        weights_per_class = {
            class_label: np.sum(kernel_weights[labels[sorted_indices] == class_label])
            for class_label in class_labels
        }

        predicted_label = max(weights_per_class, key=weights_per_class.get)
        results.append(predicted_label)

    return np.array(results)


def leave_one_out(train_data, train_labels, ks, method="custom"):
    results = {}
    n_samples = len(train_data)

    for k in ks:
        accuracies = []

        for i in range(n_samples):
            test_point = train_data[i]
            test_label = train_labels[i]

            train_subset = np.delete(train_data, i, axis=0)
            label_subset = np.delete(train_labels, i)

            if(method == "custom"):
                predicted_label = my_knn(train_subset, label_subset, [test_point], k)[0]
            elif(method == "lib"):
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(train_subset, label_subset)
                predicted_label = model.predict(test_point.reshape(1, len(test_point)))[0]
            else:
                raise Exception("Method must be lib or custom")

            accuracies.append(int(predicted_label == test_label))

        avg_loss = (len(accuracies) - sum(accuracies)) / len(accuracies)
        results[k] = avg_loss

    best_k = min(results, key=results.get)
    return best_k, results


if __name__ == "__main__":
    data, labels = read_dataset("datasets/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=25)


    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='rainbow')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig('img/dataset.png')
    plt.close()

    ks = range(1, 16)


    best_k, risks = leave_one_out(data, labels, ks, method="custom")

    plt.plot(risks.keys(), risks.values(), marker='o', label="Эмпирический риск")
    plt.axvline(best_k, color='red', linestyle='--', label=f"Лучшее k = {best_k}")
    plt.xlabel('Значения параметра k')
    plt.ylabel('Эмпирический риск')
    plt.title('Эмпирический риск для различных k (ручная реализация)')
    plt.legend()
    plt.grid()
    plt.savefig('img/my_risk.png')
    plt.close()

    k = 2
    print("\n\n\nРучная реализация при k = ", k)

    start_time = time.time()
    predictions = my_knn(X_train, y_train, X_test, k)
    end_time = time.time()

    print(classification_report(y_test, predictions))
    print("Точность ", accuracy_score(y_test, predictions))
    print(f"Время классификации (ручная реализация) {end_time - start_time:.4f} секунд")


    ks = range(1, 100)
    best_k, risks = leave_one_out(data, labels, ks, method="lib")

    plt.plot(risks.keys(), risks.values(), marker='o', label="Эмпирический риск")
    plt.axvline(best_k, color='red', linestyle='--', label=f"Лучшее k = {best_k}")
    plt.xlabel('Значения параметра k')
    plt.ylabel('Эмпирический риск')
    plt.title('Эмпирический риск для различных k (библиотечная реализация)')
    plt.legend()
    plt.grid()
    plt.savefig('img/lib_risk.png')
    plt.close()

    k = best_k
    print("\n\n\nБиблиотечная реализация при k = ", k)

    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predicted_label = model.predict(X_test)
    end_time = time.time()

    print(classification_report(y_test, predicted_label))
    print("Точность ", accuracy_score(y_test, predicted_label))
    print(f"Время классификации (библиотечная реализация) {end_time - start_time:.4f} секунд")

