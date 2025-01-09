class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None, left_proba=None):
        self.feature = feature  # индекс признака
        self.threshold = threshold  # порог для разбиения
        self.left = left  # левый дочерний узел
        self.right = right  # правый дочерний узел
        self.prediction = prediction  # предсказание для листового узла
        self.left_proba = left_proba
        self.num_elems = 0


class DecisionTreeID3Classifier:
    def __init__(self, max_depth, method, prun=False):
        self.root = None
        self.max_depth = max_depth
        self.method = method
        self.probas = {}
        self.max_proba = 0
        self.majority = None
        self.prun = prun

    def fit(self, X, y):
        for i in set(y):
            self.probas[i] = len(y.loc[y == i]) / len(y)

        self.root = self.build_tree(np.array(X), np.array(y), 0)

        self.majority = y.value_counts().argmax()

    def predict_instance(self, node, sample):
        if np.nan not in sample:
            if node.prediction is not None:
                node.num_elems += 1
                return node.prediction
            else:
                if sample[node.feature] <= node.threshold:
                    node.num_elems += 1
                    return self.predict_instance(node.left, sample)
                else:
                    node.num_elems += 1
                    return self.predict_instance(node.right, sample)
        else:
            if node.feature is None:
                return node.prediction
            else:
                return self.count_nan(node, sample, 0)

    def pruning(self, node):
        if node.num_elems == 0:
            node.prediction = self.majority
        else:
            if node.prediction is None:
                self.pruning(node.left)
                self.pruning(node.right)

    def count_nan(self, node, sample, proba):
        if node.prediction is not None:
            if self.max_proba < proba * self.probas[int(node.prediction)]:
                self.max_proba = proba * self.probas[int(node.prediction)]
                return self.probas[int(node.prediction)]
        else:
            if proba != 0:
                self.count_nan(node.left, sample, node.left_proba * proba)
                self.count_nan(node.right, sample, (1 - node.left_proba) * proba)
            else:
                self.count_nan(node.left, sample, node.left_proba)
                self.count_nan(node.right, sample, (1 - node.left_proba))

    def predict(self, X):
        pred = [self.predict_instance(self.root, sample) for sample in np.array(X)]
        if self.prun is True:
            self.pruning(self.root)
            pred = [self.predict_instance(self.root, sample) for sample in np.array(X)]
            return pred
        else:
            return pred

    @staticmethod
    def entropy(y):
        counter = Counter(y)
        total_instances = len(y)
        return -sum((count / total_instances) * math.log2(count / total_instances) for count in counter.values())

    @staticmethod
    def donskoy(y, left_mask, right_mask):
        left = {}
        right = {}
        delta = 0

        for i in set(y):
            left[i] = len(pd.Series(y[left_mask]).loc[pd.Series(y[left_mask]) == i])
            right[i] = len(pd.Series(y[right_mask]).loc[pd.Series(y[right_mask]) == i])
        if len(left.keys()) >= len(right.keys()):
            for key, value in left.items():
                if key in right.keys():
                    delta += value * right[key]
        else:
            for key, value in right.items():
                if key in left.keys():
                    delta += value * left[key]
        return delta

    def information_gain(self, y, x_column, threshold):
        left_mask = x_column <= threshold
        right_mask = x_column > threshold
        n_total = len(y)

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0, 0

        n_left = len(y[left_mask])
        n_right = len(y[right_mask])
        left_proba = n_left / n_total
        delta = 0

        if self.method == 'entropy':
            parent = self.entropy(y)
            child = (n_left / n_total) * self.entropy(y[left_mask]) + (n_right / n_total) * self.entropy(y[right_mask])
            delta = parent - child
        else:
            delta = self.donskoy(y, left_mask, right_mask)

        return delta, left_proba

    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_proba_left = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = set(X[:, feature])

            for threshold in thresholds:
                gain, left_proba = self.information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_proba_left = left_proba

        return best_feature, best_threshold, best_proba_left

    def build_tree(self, X, y, depth):
        # Если все примеры принадлежат одному классу
        if len(set(y)) == 1:
            return Node(prediction=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        # Если нет признаков для разбиения
        if X.shape[1] == 0:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        # Находим лучший признак и порог
        feature, threshold, left_proba = self.best_split(X, y)

        if feature is None:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node, left_proba=left_proba)


class DecisionTreeID3Regressor:
    def __init__(self, max_depth, prun=False):
        self.root = None
        self.max_depth = max_depth
        self.probas = {}
        self.max_proba = 0
        self.majority = None
        self.prun = prun

    def fit(self, X, y):
        for i in set(y):
            self.probas[i] = len(y.loc[y == i]) / len(y)

        self.root = self.build_tree(np.array(X), np.array(y), 0)

        self.majority = y.value_counts().argmax()

    def predict_instance(self, node, sample):
        if np.nan not in sample:
            if node.prediction is not None:
                node.num_elems += 1
                return node.prediction
            else:
                if sample[node.feature] <= node.threshold:
                    return self.predict_instance(node.left, sample)
                else:
                    return self.predict_instance(node.right, sample)
        else:
            if node.feature is None:
                return node.prediction
            else:
                return self.count_nan(node, sample, 0)

    def pruning(self, node):
        if node.num_elems == 0:
            node.prediction = self.majority
        else:
            self.pruning(node.left)
            self.pruning(node.right)

    def count_nan(self, node, sample, proba):
        if node.prediction is not None:
            if self.max_proba < proba * self.probas[int(node.prediction)]:
                self.max_proba = proba * self.probas[int(node.prediction)]
                return self.probas[int(node.prediction)]
        else:
            if proba != 0:
                self.count_nan(node.left, sample, node.left_proba * proba)
                self.count_nan(node.right, sample, (1 - node.left_proba) * proba)
            else:
                self.count_nan(node.left, sample, node.left_proba)
                self.count_nan(node.right, sample, (1 - node.left_proba))

    def predict(self, X):
        pred = [self.predict_instance(self.root, sample) for sample in np.array(X)]
        self.pruning(self.root)
        return pred

    @staticmethod
    def calculate_mse(left, right):
        total_samples = len(left) + len(right)
        mean_left = np.mean(left) if len(left) > 0 else 0
        mean_right = np.mean(right) if len(right) > 0 else 0

        mse_left = np.sum((left - mean_left) ** 2) if len(left) > 0 else 0
        mse_right = np.sum((right - mean_right) ** 2) if len(right) > 0 else 0

        return (mse_left + mse_right) / total_samples

    def information_gain(self, y, x_column, threshold):
        left_mask = x_column <= threshold
        right_mask = x_column > threshold
        n_total = len(y)

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0, 0

        n_left = len(y[left_mask])
        left_proba = n_left / n_total

        delta = self.calculate_mse(y[left_mask], y[right_mask])

        return delta, left_proba

    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_proba_left = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = set(X[:, feature])

            for threshold in thresholds:
                gain, left_proba = self.information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_proba_left = left_proba

        return best_feature, best_threshold, best_proba_left

    def build_tree(self, X, y, depth):
        # Если все принадлежат одному классу
        if len(set(y)) == 1:
            return Node(prediction=y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        # Если нет признаков для разбиения
        if X.shape[1] == 0:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        # Находим лучший признак и порог
        feature, threshold, left_proba = self.best_split(X, y)

        if feature is None:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node, left_proba=left_proba)