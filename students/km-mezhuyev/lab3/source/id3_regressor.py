import numpy as np
import pandas as pd

class ID3Regressor:
    class TreeNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, q_v=None):
            self.feature = feature # Признак, по которому происходит разбиение
            self.threshold = threshold # Порог для разбиения
            self.left = left  # Левый дочерний узел
            self.right = right  # Правый дочерний узел
            self.value = value  # Значение (если это лист)
            self.q_v = q_v  # Оценка вероятности левой ветви

        def is_leaf(self):
            return self.value is not None

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        :param max_depth: Максимальная глубина дерева
        :param min_samples_split: Минимальное количество образцов для разбиения
        :param min_samples_leaf: Минимальное количество образцов в листе
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    @staticmethod
    def mse(y):
        """
        Вычисляет среднеквадратичную ошибку - мера неопределенности.
        """
        if len(y) == 0:
            return float('inf')
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_split_mse(self, y, y_left, y_right):
        """
        Вычисляет информативность разбиения.
        """
        if len(y_left) == 0 or len(y_right) == 0:
            return float('inf')

        # Вычисляем MSE как взвешенную сумму MSE подмножеств
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)

        return left_weight * self.mse(y_left) + right_weight * self.mse(y_right)

    def _find_best_split(self, X, y):
        """
        Находит лучшее разбиение по MSE.
        """
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                current_mse = self._calculate_split_mse(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                if current_mse < best_mse:
                    best_mse = current_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Рекурсивно строит дерево регрессии.
        """
        n_samples = len(y)

        # Добавляем проверку на пустые данные
        if n_samples == 0:
            return self.TreeNode(value=0)

        # Критерии останова
        if (self.max_depth is not None and depth >= self.max_depth) or \
            n_samples < self.min_samples_split or \
            n_samples < self.min_samples_leaf or \
            len(np.unique(y)) == 1:
            return self.TreeNode(value=np.mean(y))

        # Поиск лучшего разбиения
        feature, threshold = self._find_best_split(X, y)

        # Если не нашли подходящего разбиения или MSE не улучшается
        if feature is None:
            return self.TreeNode(value=np.mean(y))

        # Разбиение данных
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Проверка на пустые разбиения
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.TreeNode(value=np.mean(y))

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Вычисление q_v
        q_v = np.sum(left_mask) / n_samples

        return self.TreeNode(
            feature=feature,
            threshold=threshold,
            left=left_tree,
            right=right_tree,
            q_v=q_v
        )

    def fit(self, X, y):
        """
        Обучает модель на данных.

        :param X: DataFrame или numpy array с признаками
        :param y: numpy array с целевыми значениями
        :return: self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.tree = self._build_tree(X, y)
        return self

    def _predict_single(self, node, x):
        """
        Предсказание для одного образца.
        """
        if node.is_leaf():
            return node.value

        if pd.isna(x[node.feature]):
            # Пропорциональное предсказание при отсутствующих значениях
            left_pred = self._predict_single(node.left, x)
            right_pred = self._predict_single(node.right, x)
            return node.q_v * left_pred + (1 - node.q_v) * right_pred

        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)

    def predict(self, X):
        """
        Предсказывает значения для набора данных.

        :param X: DataFrame или numpy array с признаками
        :return: numpy array с предсказанными значениями
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self._predict_single(self.tree, x) for x in X])

    def prune_tree(self, X_val, y_val):
        """
        Выполняет редукцию дерева на основе валидационной выборки.

        :param X_val: Валидационная выборка признаков
        :param y_val: Целевые значения валидационной выборки
        """
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()

        def _calculate_subtree_mse(node, X_subset, y_subset):
            """
            Вычисляет MSE для поддерева на подвыборке данных.
            """
            if len(X_subset) == 0:
                return float('inf')
            predictions = np.array([self._predict_single(node, x) for x in X_subset])
            return np.mean((predictions - y_subset) ** 2)

        def _calculate_leaf_mse(value, y_subset):
            """
            Вычисляет MSE для листового узла с заданным значением.
            """
            if len(y_subset) == 0:
                return float('inf')
            return np.mean((y_subset - value) ** 2)

        def _prune_node(node, X_subset, y_subset):
            """
            Выполняет редукцию для конкретного узла.
            """
            if node.is_leaf():
                return

            # Если данных нет, превращаем в лист
            if len(X_subset) == 0:
                node.value = np.mean(y_val)
                node.left = None
                node.right = None
                return

            # Вычисляем MSE текущего поддерева
            mse_before = _calculate_subtree_mse(node, X_subset, y_subset)

            # Вычисляем MSE, если превратить узел в лист
            leaf_value = np.mean(y_subset)
            mse_after = _calculate_leaf_mse(leaf_value, y_subset)

            # Если превращение в лист улучшает или не ухудшает MSE
            if mse_after <= mse_before:
                node.value = leaf_value
                node.left = None
                node.right = None
                return

            # Иначе продолжаем рекурсивно для дочерних узлов
            if not node.is_leaf():
                left_mask = X_subset[:, node.feature] <= node.threshold
                X_left, y_left = X_subset[left_mask], y_subset[left_mask]
                X_right, y_right = X_subset[~left_mask], y_subset[~left_mask]

                _prune_node(node.left, X_left, y_left)
                _prune_node(node.right, X_right, y_right)

        # Начинаем редукцию с корня
        _prune_node(self.tree, X_val, y_val)

    def get_tree_depth(self, node):
        """
        Вычисляет глубину дерева (без учета листьев).

        :param node: Узел дерева
        :return: Глубина дерева (0 для листа или пустого узла)
        """
        if node is None or node.is_leaf():
            return 0

        return 1 + max(
            self.get_tree_depth(node.left),
            self.get_tree_depth(node.right)
        )

    def count_nodes(self, node):
        """
        Подсчитывает количество узлов в дереве.
        """
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def print_tree_info(self, X_val, y_val):
        """
        Выводит информацию о дереве: глубину, количество узлов и MSE.

        :param X_val: Валидационная выборка признаков
        :param y_val: Целевые значения валидационной выборки
        """
        depth = self.get_tree_depth(self.tree)
        nodes = self.count_nodes(self.tree)
        mse = evaluate(y_val, self.predict(X_val))
        print(f"Глубина дерева: {depth}")
        print(f"Количество узлов: {nodes}")
        print(f"MSE: {mse:.4f}")

def evaluate(y_true, y_pred):
    """
    Вычисляет MSE между истинными значениями и предсказаниями.

    :param y_true: numpy array с истинными значениями
    :param y_pred: numpy array с предсказанными значениями
    :return: MSE
    """
    # Преобразуем входные данные в одномерные массивы numpy
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    # Проверяем размерности
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Размерности не совпадают: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # Обработка пропущенных значений
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        raise ValueError("Нет валидных предсказаний после удаления NaN значений")

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    return np.mean((y_true_valid - y_pred_valid) ** 2)