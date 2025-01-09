# Лабораторная работа №2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fvdm3hCCiAIhk22kvZaAnZjS_L6L3RjZ?usp=sharing)
<br>
Ссылка на датасет https://www.kaggle.com/datasets/saramah/diabets-data-set - задача предсказания диабета, бинарная классификация.

## Визуализация данных
Визуализацию данных проводим с использованием методов уменьшения размерности: t-SNE (t-distributed Stochastic Neighbor Embedding) и PCA (Principal Component Analysis). Строим диаграммы рассеивания: 
<br>
![image](https://github.com/user-attachments/assets/bc2a679c-dabe-4b13-b7d2-fa20473a9222)

## Решение
Был реализован алгоритм KNN с методом окна Парзена переменной ширины, в качестве ядра использовалось гауссово ядро, присваиваем каждому пациенту вес в зависимости от того, насколько похожи его медицинские показатели на показатели пациента, которого временно удалили. Чем больше сходство, тем больший вес получает пациент:
```
class ParzenKNN:
    def __init__(self, k, kernel='gaussian'): 
        self.k = k 
        self.kernel = kernel # тип ядра, которое будет использоваться для оценки плотности
                             # в ноутбуке также были использованы для сравнения равномерное и треугольное ядра

    def gaussian_kernel(self, distances, h):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (distances / h) ** 2)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def optimal_h(self, nearest_distances):
        # Находим h как максимальное расстояние до k-го ближайшего соседа
        return nearest_distances[self.k - 1] if self.k > 0 else 1

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = cdist([test_point], self.X, metric='euclidean').flatten() # расстояние от новой точки до всех точек в обучающем множестве с помощью евклидова метрикb
            nearest_indices = np.argsort(distances)[:self.k] # сохраняем расстояния до k ближайших соседей
            nearest_distances = distances[nearest_indices]
            nearest_labels = self.y[nearest_indices]

            # Оптимальное значение h
            h = self.optimal_h(nearest_distances)

            # Выбор ядра и рассчет весов
            if self.kernel == "gaussian":
                weights = self.gaussian_kernel(nearest_distances, h)
            elif self.kernel == "uniform":
                weights = self.uniform_kernel(nearest_distances, h)
            elif self.kernel == "triangular":
                weights = self.triangular_kernel(nearest_distances, h)
            else:
                raise ValueError("Unknown kernel type")

            label_weights = {}
            for label, weight in zip(nearest_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight

            predicted_label = max(label_weights, key=label_weights.get)
            predictions.append(predicted_label)

        return np.array(predictions)
```
### Leave One Out (LOO) 
Метод кросс-валидации "leave-one-out" (LOO) позволяет оценить качество модели, оставляя одну точку данных для проверки и тренируя модель на оставшихся данных:
```
def loo_validation(X, y, k_values):
    errors = []
    for k in k_values:
        model = ParzenKNN(k=k)
        error = 0
        for i in range(len(X)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_val = X[i].reshape(1, -1)
            y_val = y[i]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            if y_pred[0] != y_val:
                error += 1

        errors.append(error / len(X) if len(X) > 0 else 0)
    return errors
```
Эта функция оценивает, сколько раз модель ошиблась в предсказании класса оставленной точки, возвращает список ошибок для каждого значения k. И на основе ошибок строиться график эмпирического риска для различных k и выбирается оптимальный параметр k.
<br>![image](https://github.com/user-attachments/assets/9c58a424-2b28-43e9-b086-3cad22b2d4bf)
На графике очевидно, что оптимальное k = 20.

## Результаты 
### 1. Производительность алгоритмов
   - Custom ParzenKNN показал точность (accuracy) 0.75 за 0.0751 секунды. Это означает, что алгоритм правильно классифицировал 75% примеров.
   - Scikit-learn KNN продемонстрировал чуть лучшую производительность с точностью 0.79 за 0.0164 секунды. Этот результат говорит о том, что библиотека Scikit-learn реализовала KNN эффективнее, чем собственная реализация ParzenKNN.
![image](https://github.com/user-attachments/assets/c5a9ca11-4a0a-4bea-a15c-a0f11c9dfb07)

### 2. Метрики качества
   - Для обоих методов метрики точности (precision), полноты (recall) и F1-score примерно одинаковы. Это указывает на сбалансированное поведение моделей при классификации данных.
   - В случае Custom ParzenKNN, все три метрики равны 0.75, что свидетельствует об отсутствии перекоса в сторону ложноположительных или ложноотрицательных результатов.
   - У Scikit-learn KNN точность равна 0.79, а F1-score немного ниже — 0.77. Это может говорить о небольшом различии между precision и recall, но в целом модель также демонстрирует хорошую балансировку.

### 3. Влияние различных ядер на производительность Custom ParzenKNN
   - Различные ядра дали разные результаты:
     - Gaussian Kernel: точность составила 0.753.
     - Uniform Kernel: точность оказалась выше — 0.766.
     - Triangular Kernel: показал худший результат среди всех — всего 0.695.
![image](https://github.com/user-attachments/assets/1a7cf334-0f12-4d06-a9e1-39361830b9a8)
<br>Таким образом, выбор ядра имеет значительное влияние на итоговую точность модели. Uniform Kernel оказался наиболее эффективным для данной задачи.

## Заключение
На основе представленных данных можно заключить следующее:
- Scikit-learn KNN работает быстрее и точнее, чем собственная реализация ParzenKNN. Это может указывать на оптимизацию кода библиотеки или использование других техник ускорения вычислений.
- ParzenKNN показывает стабильную работу с разными ядрами, однако Uniform Kernel дал наилучшие результаты среди них.
- Метрики качества (точность, полнота, F1-score) демонстрируют, что обе реализации ведут себя сбалансированно без значительного перекоса в одну из сторон.
