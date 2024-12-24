import numpy as np
from Criterion import DonskoyCriterion, multiClassEntropyCriterion, mseCriterion


class Node:
    def __init__(self, criteria, classes=None, is_regression=False):
        self.criteria = criteria
        self.feature_idx = None
        self.beta = None
        self.information_gain = 0
        self.left_prob = 1
        self.left = None
        self.right = None
        self.classes = classes
        self.prob = None
        self.is_regression = is_regression

    def set_value(self, y):
        if self.is_regression:
            self.prob = y.mean()
        else:
            self.prob = np.array([np.sum(y == cls) for cls in self.classes]) / len(y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.set_value(y)
        for feat_idx in range(X.shape[1]):
            feature = X[:, feat_idx]
            valid_mask = ~np.isnan(feature)
            if sum(valid_mask) == 0:
                continue
            gain, beta = self.criteria(feature[valid_mask], y[valid_mask])

            if gain > self.information_gain:
                self.information_gain, self.beta, self.feature_idx = gain, beta, feat_idx

        if self.information_gain <= 0:
            return (np.array([]), np.array([])), (np.array([]), np.array([]))

        mask = X[:, self.feature_idx] <= self.beta

        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        self.left_prob = len(y_left) / len(y)

        return (X_left, y_left), (X_right, y_right)

    def predict(self, x: np.ndarray):
        if not(self.left is None and self.right is None):
            feat_value = x[self.feature_idx]
            if np.isnan(feat_value):
                left_pred = self.left.predict(x) if self.left is not None else 0
                right_pred = self.right.predict(x) if self.right is not None else 0
                return self.left_prob * left_pred + (1 - self.left_prob) * right_pred
            else:
                return self.left.predict(x) if feat_value <= self.beta else self.right.predict(x)
        return self.prob


class DecisionTreeRegression:
    def __init__(self, criterion, max_depth=4):
        self.criterion = criterion
        self.classes = None
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.tree = self._build_tree(X, y, current_depth=0)

    def _build_tree(self, X, y, current_depth):
        if len(X) == 0:
            return None
        node = Node(self.criterion, self.classes, True)
        (X_left, y_left), (X_right, y_right) = node.fit(X, y)

        if current_depth < self.max_depth and len(np.unique(y)) > 1:
            node.left = self._build_tree(X_left, y_left, current_depth + 1)
            node.right = self._build_tree(X_right, y_right, current_depth + 1)
        return node

    def predict(self, X):
        return np.array([self.tree.predict(sample) for sample in X])


class DecisionTreeClassifier:
    def __init__(self, criterion, max_depth=4):
        self.criterion = criterion
        self.classes = None
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.tree = self._build_tree(X, y, current_depth=0)

    def _build_tree(self, X, y, current_depth):
        if len(X) == 0:
            return None
        node = Node(self.criterion, self.classes, False)
        (X_left, y_left), (X_right, y_right) = node.fit(X, y)

        if current_depth < self.max_depth and len(np.unique(y)) > 1:
            node.left = self._build_tree(X_left, y_left, current_depth + 1)
            node.right = self._build_tree(X_right, y_right, current_depth + 1)
        return node

    def predict(self, X):
        return np.array([np.argmax(self.tree.predict(sample)) for sample in X])

    def prune(self, X, y):
        self.tree = self._prune_tree(self.tree, X, y)

    def _prune_tree(self, node: Node, X, y) -> Node:

        if node.left is not None or node.right is not None:
            mask_left = X[:, node.feature_idx] <= node.beta
            mask_right = ~mask_left
            if node.left is not None:
                node.left = self._prune_tree(node.left, X[mask_left], y[mask_left])
            if node.right is not None:
                node.right = self._prune_tree(node.right, X[mask_right], y[mask_right])
        errors = compute_errors(node, X, y)
        min_err_idx = np.argmin(errors)


        if min_err_idx == 0:
            pass
        elif min_err_idx == 1:
            node = node.left
        elif min_err_idx == 2:
            node = node.right
        else:
            node.set_value(y)
            node.right = None
            node.left = None
        return node

def compute_errors(node, X, y):
    if node.left is None and node.right is None:
        unique_classes, counts = np.unique(y, return_counts=True)
        if counts.size != 0:
            most_freq_class = unique_classes[np.argmax(counts)]
            base_err = np.mean(y != most_freq_class)
        else:
            base_err = float('inf')
        predicted_class = np.argmax(node.prob)
        return (
            np.mean(y != predicted_class),
            float('inf'),
            float('inf'),
            base_err
        )
    predictions = np.array([np.argmax(node.predict(row)) for row in X])
    err_curr = np.mean(y != predictions)

    if node.left is not None:
        left_predictions = np.array([np.argmax(node.left.predict(row)) for row in X])
        err_left = np.mean(y != left_predictions)
    else:
        err_left = float('inf')

    if node.right is not None:
        right_predictions = np.array([np.argmax(node.right.predict(row)) for row in X])
        err_right = np.mean(y != right_predictions)
    else:
        err_right = float('inf')

    unique_classes, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        err_base = float('inf')
    else:
        most_freq_class = unique_classes[np.argmax(counts)]
        err_base = np.mean(y != most_freq_class)

    return err_curr, err_left, err_right, err_base

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from load_data import load_scaled_data
    from sklearn.metrics import accuracy_score, mean_squared_error
    from Criterion import Criterions

    df = load_scaled_data()
    # Разделение на X и y
    X = np.array(df.drop(columns=['Survived']))
    y = np.array(df['Survived'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier(Criterions.multiClassEntropy, max_depth=7)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)
    print("Predictions:", predictions)
    print("True labels:", y_val)
    print("Accuracy:", accuracy_score(y_val, predictions))

    tree.prune(X_val, y_val)
    predictions2 = tree.predict(X_val).round()
    print("Predictions:", predictions2)
    print("Accuracy:", accuracy_score(y_val, predictions2))

    tree2 = DecisionTreeClassifier(Criterions.donskoy, max_depth=4)
    tree2.fit(X_train, y_train)
    predictions3 = tree2.predict(X_val)
    print("Predictions:", predictions3)
    print("Accuracy:", accuracy_score(y_val, predictions3))

    X = np.array(df.drop(columns=['Fare']))
    y = np.array(df['Fare'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeRegression(Criterions.mse, max_depth=4)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_val)
    print("Predictions:", predictions)
    print("True labels:", y_val)
    print("Accuracy:", mean_squared_error(y_val, predictions))

