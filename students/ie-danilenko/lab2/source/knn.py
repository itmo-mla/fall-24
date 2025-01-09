from read_fish import read_fish
import numpy as np

def knn(X, y, u, k):
    predictions = []
    unique_classes = np.unique(y)
    for x in u:
        distances = np.linalg.norm(x - X, axis=1)
        sorted_index = np.argsort(distances)
        distances = distances[sorted_index]
        weights = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distances / distances[k + 1]) ** 2)
        class_weights = {cls: np.sum(weights[y[sorted_index] == cls]) for cls in unique_classes}
        predicted_class = max(class_weights, key=class_weights.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

if __name__ == "__main__":
    X, y = read_fish('dataset/fish.csv')

    test_i = 1000
    X = np.delete(X, test_i, axis=0)
    y = np.delete(y, test_i)
    u = X[test_i]
    u = np.expand_dims(u, axis=0)
    predicted_classes = knn(X, y, u, 1)
    print(f"Истинный класс: {y[test_i]}")
    print(f"Предсказанный класс: {predicted_classes[0]}")