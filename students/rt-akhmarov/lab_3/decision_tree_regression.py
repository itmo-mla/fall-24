import numpy as np

class Node:
    """
    Узел дерева решений для задачи регрессии.
    """
    def __init__(self, criteria):
        self.criteria = criteria  # Функция, вычисляющая прирост информации и порог разделения
        self.predicat = None  # Порог разделения для текущего узла
        self.feature_idx = None  # Индекс признака, по которому производится разделение
        self.info_gain = 0  # Максимальный прирост информации
        self.value = None  # Среднее значение целевой переменной в листе
        self.left = None  # Ссылка на левый дочерний узел
        self.right = None  # Ссылка на правый дочерний узел

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Подбирает оптимальное разделение для узла.
        """
        # Если текущий узел является листом, сохраняем среднее значение целевой переменной
        self.value = np.mean(y)

        # Перебираем все признаки для поиска наилучшего разделения
        for idx, feature in enumerate(X.T):  # Перебираем каждый столбец (признак) в X
            feature_no_nan = feature[~np.isnan(feature)]  # Удаляем значения NaN из признака
            y_no_nan = y[~np.isnan(feature)]  # Удаляем соответствующие значения целевой переменной
            if len(y_no_nan) == 0:  # Если после удаления NaN данных не осталось, пропускаем этот признак
                continue

            # Применяем критерий для вычисления прироста информации и порога
            info_gain, predicat = self.criteria(feature_no_nan, y_no_nan)

            # Если найденный прирост информации лучше текущего, обновляем параметры узла
            if info_gain > self.info_gain:
                self.info_gain = info_gain
                self.predicat = predicat
                self.feature_idx = idx

        # Если прирост информации незначителен, возвращаем пустые массивы
        if self.info_gain <= 0 or self.info_gain is None:
            return (np.array([]), np.array([])), (np.array([]), np.array([]))

        # Разделяем данные по найденному признаку и порогу
        feature = X[:, self.feature_idx]
        mask_left = (feature <= self.predicat) & ~np.isnan(feature)  # Левые данные (<= порога)
        mask_right = (feature > self.predicat) & ~np.isnan(feature)  # Правые данные (> порога)

        X_left = X[mask_left]  # Левая часть признаков
        y_left = y[mask_left]  # Целевая переменная для левой части

        X_right = X[mask_right]  # Правая часть признаков
        y_right = y[mask_right]  # Целевая переменная для правой части

        return (X_left, y_left), (X_right, y_right)

    def predict_single(self, x: np.ndarray):
        """
        Делает предсказание для одного экземпляра данных.
        """
        # Если узел листовой, возвращаем среднее значение целевой переменной
        if self.left is None and self.right is None:
            return self.value

        val = x[self.feature_idx]  # Значение признака, по которому разделяем

        if np.isnan(val):  # Если значение NaN, возвращаем значение текущего узла
            return self.value
        else:
            # Направляем влево или вправо в зависимости от порога
            if val <= self.predicat:
                return self.left.predict_single(x)
            else:
                return self.right.predict_single(x)

    def predict(self, X: np.ndarray):
        """
        Делает предсказания для набора данных.
        """
        return np.array([self.predict_single(row) for row in X])

def decision_tree_regressor(X, y, criteria, current_depth=0, max_depth=4):
    """
    Построение дерева решений для задачи регрессии.
    """
    node = Node(criteria=criteria)  # Создаем новый узел

    (X_left, y_left), (X_right, y_right) = node.fit(X, y)  # Пытаемся разделить данные

    # Рекурсивно строим дерево, если не достигнута максимальная глубина и данных достаточно для разбиения
    if current_depth < max_depth and len(y) > 1:
        current_depth += 1
        if len(y_left) > 0:
            node.left = decision_tree_regressor(X_left, y_left, criteria, current_depth, max_depth)
        if len(y_right) > 0:
            node.right = decision_tree_regressor(X_right, y_right, criteria, current_depth, max_depth)

    return node
