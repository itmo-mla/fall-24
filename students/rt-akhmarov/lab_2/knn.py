import numpy as np

# Ядро Парзена (Гауссовское ядро)
def gaussian_kernel(distance, bandwidth):   
    return (1 / (np.sqrt(2 * np.pi) * bandwidth)) * np.exp(-0.5 * (distance / bandwidth) ** 2)

# KNN с окном Парзена переменной ширины
def knn_parzen(X_train, y_train, X_test, k, bandwidth):
    predictions = []
    
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)  # Расстояния до всех точек обучения
        neighbors_idx = np.argsort(distances)[:k+1]  # Индексы k ближайших соседей
        if bandwidth is None:
            bandwidth = distances[neighbors_idx[-1]]
            # print(f'non fixed bandwidth: {bandwidth}')

        neighbors_idx = neighbors_idx[:-1]
        weights = gaussian_kernel(distances[neighbors_idx], bandwidth)  # Вычисляем веса с помощью ядра
        classes = y_train[neighbors_idx]  # Классы соседей
        
        # Взвешенное голосование
        class_weights = {}
        for cls, w in zip(classes, weights):
            if cls not in class_weights:
                class_weights[cls] = 0
            class_weights[cls] += w
        
        # Предсказываем класс с максимальным взвешенным голосом
        predictions.append(max(class_weights, key=class_weights.get))
    
    return np.array(predictions)

# Подбор оптимального k методом Leave-One-Out (LOO)
def loo_knn_parzen(X, y, bandwidth):
    n = len(X)
    best_k = 1
    min_risk = n
    risks = []

    for k in range(1, n):
        risk = 0
        
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i].reshape(1, -1)
            y_test = y[i]
            
            prediction = knn_parzen(X_train, y_train, X_test, k, bandwidth)
            if prediction != y_test:
                risk += 1
        
        risk = risk / n
        if risk < min_risk:
            min_risk = risk
            best_k = k

        risks.append(risk)
    return best_k, risks
