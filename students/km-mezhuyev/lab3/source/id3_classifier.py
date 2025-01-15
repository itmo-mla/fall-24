import numpy as np
import pandas as pd

class ID3Classifier:
    class TreeNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, q_v=None):
            self.feature = feature  # Признак, по которому происходит разбиение
            self.threshold = threshold  # Порог для разбиения
            self.left = left  # Левый дочерний узел
            self.right = right  # Правый дочерний узел
            self.value = value  # Значение (если это лист)
            self.q_v = q_v  # Оценка вероятности левой ветви

        def is_leaf(self):
            return self.value is not None

    def __init__(self, criterion_type='entropy', max_depth=None):
        """
        : param criterion_type: Тип критерия качества разбиения ('entropy' или 'donsky')
        : param max_depth: Максимальная глубина дерева как критерий останова
        """
        self.tree = None
        self.V_inner = set()  # Множество внутренних вершин
        self.V_leaf = set()   # Множество листовых вершин
        self.criterion_type = criterion_type
        self.max_depth = max_depth  # Максимальная глубина дерева как критерий останова

    @staticmethod
    def multiclass_entropy(y):
        """
        Вычисляет энтропию для мультиклассовой задачи.

        :param y: Массив меток классов
        :return: Энтропия
        """
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def donskoy_criterion(y, left_mask, right_mask):
        """
        Вычисляет критерий Донского для оценки качества разбиения.

        :param y: Массив меток классов
        :param left_mask: Маска для левой ветви
        :param right_mask: Маска для правой ветви
        :return: Значение критерия Донского
        """
        # Получаем индексы объектов в левом и правом поддереве
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        count = 0

        # Сравниваем каждый объект из левого поддерева с каждым объектом из правого поддерева
        for i in left_indices:
            for j in right_indices:
                # Если метки классов разные, увеличиваем счетчик
                if y[i] != y[j]:
                    count += 1

        return count

    def fit(self, X, y):
        """
        Обучает модель на данных.
        :param X: Данные в виде DataFrame
        :param y: Массив меток классов в виде Series или списка
        """
        # Преобразуем входные данные в numpy массивы
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()  # Сохраняем названия признаков
            self.feature_indices = {i: name for i, name in enumerate(self.feature_names)}  # Словарь индексов и названий
            X = X.to_numpy()
        else:
            raise ValueError("X должен быть DataFrame")

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, list):
            y = np.array(y)
        else:
            raise ValueError("y должен быть массивом Series или списком")

        features = list(range(X.shape[1]))  # Индексы признаков
        self.tree = self._learn_id3(X, y, features)

    def _learn_id3(self, U, y, features, depth=0):
        """
        Реализует алгоритм LearnID3 для построения дерева решений.
        """
        features = list(range(U.shape[1]))

        # Если все объекты из U лежат в одном классе
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            node = self.TreeNode(value=unique_classes[0])
            self.V_leaf.add(node)
            return node

        # Проверка на максимальную глубину
        if self.max_depth is not None and depth >= self.max_depth:
            major_class = self._majority_class(y)
            node = self.TreeNode(value=major_class)
            self.V_leaf.add(node)
            return node

        # Находим предикат с максимальной информативностью
        best_j, best_t = self._find_best_split(U, y, features)

        # Учитываем только объекты с определенными значениями для разбиения
        valid_mask = ~np.isnan(U[:, best_j])
        valid_data = U[valid_mask]
        valid_y = y[valid_mask]

        left_mask = valid_data[:, best_j] <= best_t
        U0, y0 = valid_data[left_mask], valid_y[left_mask]
        U1, y1 = valid_data[~left_mask], valid_y[~left_mask]

        # Если одно из множеств пусто, создаем лист с мажоритарным классом
        if len(U0) == 0 or len(U1) == 0:
            major_class = self._majority_class(y)
            node = self.TreeNode(value=major_class)
            self.V_leaf.add(node)
            return node

        # Вычисляем q_v - оценку вероятности левой ветви
        q_v = len(U0) / len(valid_data)

        # Создаем новую внутреннюю вершину
        node = self.TreeNode(
            feature=best_j,
            threshold=best_t,
            q_v=q_v
        )
        self.V_inner.add(node)

        # Рекурсивно строим поддеревья, увеличивая глубину
        node.left = self._learn_id3(U0, y0, features, depth + 1)
        node.right = self._learn_id3(U1, y1, features, depth + 1)

        return node

    def _majority_class(self, y):
        """
        Возвращает мажоритарный класс в выборке.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def _find_best_split(self, U, y, features):
        """
        Находит лучший предикат для разбиения.
        """
        best_gain = -float('inf')
        best_j = features[0] # Инициализируем первым признаком
        best_t = None

        for j in features:
            thresholds = np.unique(U[:, j])
            for t in thresholds:
                gain = self._information_gain(y, U[:, j], t)

                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_t = t

        return best_j, best_t

    def _information_gain(self, y, x_column, threshold):
        """
        Вычисляет информационный выигрыш для данного разбиения.
        """
        left_mask = x_column <= threshold
        right_mask = x_column > threshold
        n_total = len(y)

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        n_left = len(y[left_mask])
        n_right = len(y[right_mask])
        delta = 0

        if self.criterion_type == 'entropy':
            s0 = self.multiclass_entropy(y)
            s1_s2 = (n_left / n_total) * self.multiclass_entropy(y[left_mask]) + \
                    (n_right / n_total) * self.multiclass_entropy(y[right_mask])
            delta = s0 - s1_s2
            return delta
        elif self.criterion_type == 'donsky':
            return self.donskoy_criterion(y, left_mask, right_mask)
        else:
            raise ValueError(f"Неизвестный тип критерия: {self.criterion_type}")

    def predict(self, X, y=None):
        """
        Предсказывает метки для набора данных и вычисляет точность, если предоставлены истинные метки.

        :param X: DataFrame с признаками
        :param y: Истинные метки (опционально)
        :return: Если y не None, возвращает (predictions, accuracy), иначе только predictions
        """
        predictions = []
        for _, sample in X.iterrows():
            proba = self._predict_proba(self.tree, sample)
            if isinstance(proba, dict):
                predictions.append(max(proba.items(), key=lambda x: x[1])[0])
            else:
                predictions.append(proba)
        if y is not None:
            # Вычисляем точность
            accuracy = np.mean((predictions) == y)
            return predictions, accuracy
        return predictions

    def _predict_proba(self, node, sample):
        """
        Вычисляет вероятности классов для одного образца.

        :param node: Узел дерева
        :param sample: Образец для предсказания
        :return: Вероятности классов
        """
        if node.is_leaf():
            return node.value

        # Если значение признака не определено
        if pd.isna(sample[node.feature]):
            # Пропорциональное распределение
            left_proba = self._predict_proba(node.left, sample)
            right_proba = self._predict_proba(node.right, sample)
            return node.q_v * left_proba + (1 - node.q_v) * right_proba

        # Если значение признака определено
        if sample[node.feature] <= node.threshold:
            return self._predict_proba(node.left, sample)
        else:
            return self._predict_proba(node.right, sample)

    def _predict_one(self, node, sample):
        """
        Предсказывает метку для одного образца данных.

        :param node: Узел дерева
        :param sample: Образец для предсказания
        :return: Предсказанная метка
        """
        if node.is_leaf():
            return node.value

        # Если значение признака не определено
        if pd.isna(sample[node.feature]):
            # Пропорциональное распределение
            left_proba = self._predict_proba(node.left, sample)
            right_proba = self._predict_proba(node.right, sample)
            return node.q_v * left_proba + (1 - node.q_v) * right_proba

        # Проверка типов данных
        feature_value = sample[node.feature]
        threshold_value = node.threshold

        if isinstance(feature_value, (int, float)) and isinstance(threshold_value, (int, float)):
            if feature_value <= threshold_value:
                return self._predict_one(node.left, sample)
            else:
                return self._predict_one(node.right, sample)
        else:
            raise ValueError("Тип данных признака и порога должны совпадать.")

    def visualize_tree(self):
        """
        Визуализирует структуру дерева решений в более читаемом формате
        """
        def get_human_readable_condition(feature, threshold, direction):
            """
            Преобразует условие в читаемый вид
            """
            feature_name = self.feature_indices[feature]

            # Специальные правила для разных признаков
            if feature_name == 'Sex':
                return f"{'Мужчина' if direction == 'right' else 'Женщина'}"
            elif feature_name == 'Age':
                return f"Возраст {'>=' if direction == 'right' else '<='} {threshold:.1f} лет"
            elif feature_name == 'Pclass':
                return f"Номер класса {'>' if direction == 'right' else '<='} {int(threshold)}"
            elif feature_name == 'Fare':
                return f"Стоимость билета {'>' if direction == 'right' else '<='} {threshold:.2f}$"
            elif feature_name == 'SibSp':
                return f"Количество членов семьи {'>' if direction == 'right' else '<='} {int(threshold)}"
            else:
                return f"{feature_name} {'>' if direction == 'right' else '<='} {threshold}"

        def print_tree(node, depth=0, prefix="└── ", direction=None):
            indent = "    " * depth
    
            if node.is_leaf():
                survival = "Выжил" if node.value == 1 else "Не выжил"
                print(f"{indent}{prefix}{survival}")
                return
            
            # Печать условия для корня
            if depth == 0:
                print("Дерево решений для предсказания выживаемости на Титанике:")
                print(f"├── {get_human_readable_condition(node.feature, node.threshold, 'left')}")
            
            # Рекурсивный вывод левой и правой ветви
            if node.left:
                condition = get_human_readable_condition(node.feature, node.threshold, 'left')
                if not node.left.is_leaf():
                    print(f"{indent}├── Если {condition}:")
                print_tree(node.left, depth + 1, "├── " if node.right else "└── ", 'left')
            
            if node.right:
                condition = get_human_readable_condition(node.feature, node.threshold, 'right')
                if not node.right.is_leaf():
                    print(f"{indent}└── Если {condition}:")
                print_tree(node.right, depth + 1, "└── ", 'right')

        print_tree(self.tree)

    def _count_errors(self, node, X_subset, y_subset):
        """
        Подсчитывает количество ошибок для поддерева.
        """
        if len(X_subset) == 0:
            return 0

        predictions = [self._predict_one(node, x) for x in X_subset]
        return sum(pred != true for pred, true in zip(predictions, y_subset))

    def prune_tree(self, X_val, y_val):
        """
        Выполняет редукцию дерева на основе контрольной выборки.

        :param X_val: Контрольная выборка признаков
        :param y_val: Метки классов контрольной выборки
        """
        # Преобразуем входные данные в numpy массивы
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()

        def _prune_node(node, X_subset, y_subset):
            """
            Выполняет редукцию для конкретного узла.
            """
            if node.is_leaf():
                return

            # Получаем объекты, дошедшие до текущей вершины
            if len(X_subset) == 0:
                node.value = self._majority_class(y_val)
                node.left = None
                node.right = None
                return

            # Считаем ошибки для четырех вариантов
            r_v = self._count_errors(node, X_subset, y_subset)  # Текущее поддерево

            # Создаем временные листовые узлы для левого и правого поддерева
            left_leaf = self.TreeNode(value=self._majority_class(y_subset[X_subset[:, node.feature] <= node.threshold]))
            right_leaf = self.TreeNode(value=self._majority_class(y_subset[X_subset[:, node.feature] > node.threshold]))

            # Ошибки для левого и правого поддерева
            r_L = self._count_errors(left_leaf, X_subset, y_subset)
            r_R = self._count_errors(right_leaf, X_subset, y_subset)

            # Ошибки для каждого класса
            r_c = {}
            for c in np.unique(y_val):
                leaf_node = self.TreeNode(value=c)
                r_c[c] = self._count_errors(leaf_node, X_subset, y_subset)

            # Находим минимальное количество ошибок
            min_errors = min(r_v, r_L, r_R, min(r_c.values()))
            
            # Применяем редукцию в зависимости от минимума ошибок
            if min_errors == r_v:
                # Сохраняем текущее поддерево
                _prune_node(node.left, X_subset[X_subset[:, node.feature] <= node.threshold],
                          y_subset[X_subset[:, node.feature] <= node.threshold])
                _prune_node(node.right, X_subset[X_subset[:, node.feature] > node.threshold],
                          y_subset[X_subset[:, node.feature] > node.threshold])
            elif min_errors == r_L:
                # Заменяем на левое поддерево
                node.value = left_leaf.value
                node.left = None
                node.right = None
            elif min_errors == r_R:
                # Заменяем на правое поддерево
                node.value = right_leaf.value
                node.left = None
                node.right = None
            else:
                # Заменяем листом с оптимальным классом
                best_class = min(r_c.items(), key=lambda x: x[1])[0]
                node.value = best_class
                node.left = None
                node.right = None

        # Начинаем редукцию с корня
        _prune_node(self.tree, X_val, y_val)