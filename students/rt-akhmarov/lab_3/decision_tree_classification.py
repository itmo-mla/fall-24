import numpy as np

class Node:
    # Конструктор класса Node. Каждый узел содержит информацию о критериях, классах, индексе признака,
    # приросте информации, вероятности разделения на левую ветвь и ссылки на дочерние узлы.
    def __init__(self, criteria, classes=None):
        self.criteria = criteria  # Функция, вычисляющая прирост информации и порог разделения
        self.predicat = None  # Порог разделения для текущего узла
        self.feature_idx = None  # Индекс признака, по которому производится разделение
        self.info_gain = 0  # Максимальный прирост информации
        self.prob_left = 1  # Вероятность, с которой данные принадлежат левой ветви
        self.left = None  # Ссылка на левый дочерний узел
        self.right = None  # Ссылка на правый дочерний узел
        self.classes = classes  # Список всех возможных классов
        self.prob = None  # Распределение вероятностей классов в текущем узле

    # Метод подбора оптимального разделения на основе входных данных X и меток классов y.
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Вычисляем вероятность каждого класса в текущем узле
        self.prob = np.array([np.sum(y == cls) for cls in self.classes]) / len(y)

        # Перебираем все признаки для поиска наилучшего разделения
        for (idx, feature) in enumerate(X.T):  # Перебираем каждый столбец (признак) в X
            feature_no_nan = feature[~np.isnan(feature)]  # Удаляем значения NaN из признака
            y_no_nan = y[~np.isnan(feature)]  # Удаляем соответствующие метки классов
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
        y_left = y[mask_left]  # Метки классов для левой части

        X_right = X[mask_right]  # Правая часть признаков
        y_right = y[mask_right]  # Метки классов для правой части

        # Рассчитываем вероятность данных для левой ветви
        total_split = len(X_left) + len(X_right)
        self.prob_left = len(X_left) / total_split

        return (X_left, y_left), (X_right, y_right)

    # Предсказание для одного экземпляра данных
    def predict_single(self, x: np.ndarray):
        if self.left is None and self.right is None:  # Если узел листовой, возвращаем вероятности классов
            return self.prob

        val = x[self.feature_idx]  # Значение признака, по которому разделяем

        if np.isnan(val):  # Если значение NaN, обрабатываем случай отдельно
            if self.left is None and self.right is None:
                return self.prob  # Если дочерние узлы отсутствуют, возвращаем текущие вероятности
            elif self.left is not None and self.right is None:
                return self.left.predict_single(x)  # Если только левая ветвь, идем влево
            elif self.left is None and self.right is not None:
                return self.right.predict_single(x)  # Если только правая ветвь, идем вправо
            else:
                # Если обе ветви существуют, комбинируем вероятности обеих ветвей
                left_proba = self.left.predict_single(x)
                right_proba = self.right.predict_single(x)
                return self.prob_left * left_proba + (1 - self.prob_left) * right_proba
        else:
            # Направляем влево или вправо в зависимости от порога
            if val <= self.predicat:
                return self.left.predict_single(x)
            else:
                return self.right.predict_single(x)

    # Предсказание для набора данных
    def predict(self, X: np.ndarray):
        return np.array([self.predict_single(row) for row in X])
    

# Функция построения дерева решений
def decision_tree_classifier(X, y, criteria, classes, current_depth=0, max_depth=4):
    node = Node(criteria=criteria, classes=classes)  # Создаем новый узел

    (X_left, y_left), (X_right, y_right) = node.fit(X, y)  # Пытаемся разделить данные

    # Рекурсивно строим дерево, если не достигнута максимальная глубина и классы не уникальны
    if current_depth < max_depth and len(np.unique(y)) > 1:
        current_depth += 1
        # Рекурсивно создаем левую ветвь
        node.left = decision_tree_classifier(X_left, y_left, criteria, classes, current_depth, max_depth)
        # Рекурсивно создаем правую ветвь
        node.right = decision_tree_classifier(X_right, y_right, criteria, classes, current_depth, max_depth)

    return node  # Возвращаем узел
