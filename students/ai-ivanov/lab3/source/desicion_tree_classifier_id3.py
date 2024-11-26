from typing import Any, Literal
import numpy as np
from pydantic import BaseModel


class Node(BaseModel):
    kind: Literal["parent", "leaf"]


class ParentNode(Node):
    kind: Literal["parent"] = "parent"
    feature: int
    threshold: float | str
    left: Node
    right: Node


class LeafNode(Node):
    kind: Literal["leaf"] = "leaf"
    class_: int


class DecisionTreeClassifierID3:
    def __init__(
        self,
        criterion: Literal["entropy", "donskoy"] = "entropy",
        min_samples_split: int = 2,
        max_depth: int = 5,
    ):
        """
        Initialize ID3 decision tree classifier

        Parameters:
        -----------
        criterion : str, default='entropy'
            The function to measure the quality of a split.
            Supported criteria are 'entropy' for information gain with entropy
            and 'donskoy' for Donskoy's criterion
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node
        max_depth : int, default=5
            The maximum depth of the tree
        """
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
        self._explicit_predict = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
    ):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).
        feature_types : list[Literal["categorical", "numeric"]]
            The types of features.
        """
        self.tree = self._build_tree(X, y, feature_types)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
        depth: int = 0,
    ) -> Node:
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # Base cases
        if n_classes == 1:
            return LeafNode(class_=y[0])
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return LeafNode(class_=np.argmax(np.bincount(y)))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, feature_types)

        # Split the data
        match best_threshold:
            case float():
                # Numeric feature
                left_mask = X[:, best_feature] <= best_threshold
                right_mask = ~left_mask
            case _:
                # Categorical feature
                left_mask = X[:, best_feature] == best_threshold
                right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(
            X[left_mask], y[left_mask], feature_types, depth + 1
        )
        right_subtree = self._build_tree(
            X[right_mask], y[right_mask], feature_types, depth + 1
        )

        return ParentNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
    ) -> tuple[int, float | str]:
        _, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            if feature_types[feature] == "categorical":
                unique_values = list(set(X[:, feature]))
                # Try each category as a binary split
                for category in unique_values:
                    gain = self._information_gain(X[:, feature], y, category)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = category
            else:
                unique_values = np.unique(X[:, feature])
                # Numeric feature - use threshold splits
                for threshold in unique_values:
                    gain = self._information_gain(X[:, feature], y, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(
        self,
        X_column: np.ndarray,
        y: np.ndarray,
        threshold: float | Any,
    ) -> float:
        # Split the data based on threshold
        match threshold:
            case float() if np.isnan(threshold):
                # If threshold is NaN, return 0 gain
                return 0.0
            case float():
                try:
                    left_mask = X_column <= threshold
                    right_mask = ~left_mask
                except TypeError:
                    raise ValueError(
                        f"Threshold must be a number for numeric features, got {threshold}: {type(threshold)}"
                    )
            case _:
                left_mask = X_column == threshold
                right_mask = ~left_mask

        # Get the child node samples
        left_y = y[left_mask]
        right_y = y[right_mask]

        if self.criterion == "entropy":
            # Calculate parent entropy
            parent_entropy = self._entropy(y)

            # Calculate weights of splits
            n_samples = len(y)
            w_left = len(left_y) / n_samples
            w_right = len(right_y) / n_samples

            # Calculate weighted child entropy
            child_entropy = w_left * self._entropy(left_y) + w_right * self._entropy(
                right_y
            )
            return parent_entropy - child_entropy

        else:  # donskoy criterion
            n_total = len(y)
            n_left = len(left_y)
            n_right = len(right_y)

            # Normalization factor
            norm_factor = (n_left * n_right) / (n_total**2)

            # Calculate class proportions difference
            total_diff = 0
            for class_label in np.unique(y):
                left_prop = np.sum(left_y == class_label) / n_left if n_left > 0 else 0
                right_prop = (
                    np.sum(right_y == class_label) / n_right if n_right > 0 else 0
                )
                total_diff += abs(left_prop - right_prop)

            return norm_factor * total_diff

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node."""
        # Handle empty arrays
        if len(y) == 0:
            return 0.0

        # Calculate probabilities of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Calculate entropy using formula: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def prune(self, X_test: np.ndarray, y_test: np.ndarray):
        """Prune the tree using test data and Reduced Error Pruning."""

        if self.tree is None:
            raise ValueError("Tree must be fitted before pruning")

        self.tree = self._prune_recursive(self.tree, X_test, y_test)

    def _evaluate_accuracy(self, node: Node, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy for the given node."""
        predictions = []
        for row in X:
            current = node
            while True:
                match current:
                    # Navigate until we hit a leaf
                    case LeafNode(class_=class_):
                        predictions.append(class_)
                        break

                    # Numeric feature
                    case ParentNode(
                        threshold=float() as threshold,
                        feature=feature,
                        left=left,
                        right=right,
                    ):
                        go_left = row[feature] <= threshold
                        current = left if go_left else right

                    # Categorical feature
                    case ParentNode(
                        threshold=threshold,
                        feature=feature,
                        left=left,
                        right=right,
                    ):
                        go_left = row[feature] == threshold
                        current = left if go_left else right
                    case _:
                        raise ValueError(f"Invalid node: {current}")

        return np.mean(predictions == y)

    def _should_prune(
        self, node: Node, X: np.ndarray, y: np.ndarray
    ) -> tuple[bool, float]:
        """Determine if node should be pruned by comparing accuracies."""
        accuracy_before = self._evaluate_accuracy(node, X, y)

        temp_node = LeafNode(class_=np.argmax(np.bincount(y)))

        accuracy_after = self._evaluate_accuracy(temp_node, X, y)

        return accuracy_after >= accuracy_before, accuracy_after

    def _prune_recursive(self, node: Node, X: np.ndarray, y: np.ndarray) -> Node:
        """Recursively prune the tree."""
        match node:
            # Base case: if we're at a leaf, return
            case LeafNode():
                return node

            case ParentNode(
                threshold=threshold,
                feature=feature,
                left=left,
                right=right,
            ):
                # Recursively prune children first
                match threshold:
                    case float():
                        left_mask = X[:, feature] <= threshold
                    case _:
                        left_mask = X[:, feature] == threshold

                right_mask = ~left_mask

                if len(X[left_mask]) > 0:
                    node.left = self._prune_recursive(left, X[left_mask], y[left_mask])
                if len(X[right_mask]) > 0:
                    node.right = self._prune_recursive(
                        right, X[right_mask], y[right_mask]
                    )

                # After pruning children, check if this node should be pruned
                should_prune, _ = self._should_prune(node, X, y)
                if should_prune:
                    return LeafNode(class_=np.argmax(np.bincount(y)))

                return node
            case _:
                raise ValueError(f"Invalid node: {node}")

    def __str__(self) -> str:
        """Return string representation of the tree."""

        if not self.tree:
            return "Tree not fitted yet"

        def _to_str(node: Node, level: int = 0) -> str:
            indent = "\t" * level

            match node:
                case LeafNode(class_=class_):
                    return f"{indent}return {class_}"
                case ParentNode(
                    threshold=threshold,
                    feature=feature,
                    left=left,
                    right=right,
                ):
                    left_str = _to_str(left, level + 1)
                    right_str = _to_str(right, level + 1)

                    match threshold:
                        case float():
                            sign = "<="
                        case _:
                            sign = "=="
                            threshold = f"'{threshold}'"

                    return (
                        f"{indent}if feature[{feature}] {sign} {threshold}:\n"
                        f"{left_str}\n"
                        f"{indent}else:\n"
                        f"{right_str}"
                    )

        return f"def explicit_predict(feature):\n{_to_str(self.tree, 1)}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X."""
        return np.array([self._predict_row(row) for row in X])

    def predict_using_exec(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X using exec."""
        return np.array([self._predict_row_using_exec(row) for row in X])

    def _predict_row(self, row: np.ndarray) -> int:
        """Predict class for a single row."""
        node = self.tree
        while True:
            match node:
                case LeafNode(class_=class_):
                    return class_
                case ParentNode(
                    threshold=threshold,
                    feature=feature,
                    left=left,
                    right=right,
                ):
                    match threshold:
                        case float():
                            go_left = row[feature] <= threshold
                        case _:
                            go_left = row[feature] == threshold

                    if go_left:
                        node = left
                    else:
                        node = right
                case _:
                    raise ValueError(f"Invalid node: {node}")

    def _predict_row_using_exec(self, feature: np.ndarray) -> int:
        """Predict class for a single row using exec."""
        if not self._explicit_predict:
            declare_predict = self.__str__()
            namespace = {}

            exec(declare_predict, namespace)
            self._explicit_predict = namespace["explicit_predict"]

        return self._explicit_predict(feature)
