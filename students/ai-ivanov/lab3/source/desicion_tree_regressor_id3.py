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
    value: float


class DecisionTreeRegressorID3:
    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 5,
    ):
        """
        Initialize ID3 decision tree regressor

        Parameters:
        -----------
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node
        max_depth : int, default=5
            The maximum depth of the tree
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: list[Literal["categorical", "numeric"]],
    ):
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
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

        # Base cases
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return LeafNode(value=np.mean(y))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, feature_types)

        # If no good split is found, return leaf
        if best_feature is None:
            return LeafNode(value=np.mean(y, where=~np.isnan(y)))

        # Split the data
        match best_threshold:
            case float():
                left_mask = X[:, best_feature] <= best_threshold
                right_mask = ~left_mask
            case _:
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
    ) -> tuple[int | None, float | str | None]:
        _, n_features = X.shape
        best_mse = np.inf
        best_feature = None
        best_threshold = None

        current_mse = np.mean((y - np.mean(y, where=~np.isnan(y))) ** 2, where=~np.isnan(y))

        for feature in range(n_features):
            if feature_types[feature] == "categorical":
                unique_values = list(set(X[:, feature]))
                for category in unique_values:
                    mse = self._calculate_mse(X[:, feature], y, category)
                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = category
            else:
                # Sort unique values for numeric features
                unique_values = np.sort(np.unique(X[:, feature]))
                for threshold in unique_values:
                    if np.isnan(threshold):
                        continue
                    mse = self._calculate_mse(X[:, feature], y, threshold)
                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold
        # If no split improves MSE, return None
        if best_mse >= current_mse:
            return None, None

        return best_feature, best_threshold

    def _calculate_mse(
        self,
        X_column: np.ndarray,
        y: np.ndarray,
        threshold: float | Any,
    ) -> float:
        # Split the data based on threshold
        match threshold:
            case float() if np.isnan(threshold):
                return np.inf
            case float():
                left_mask = X_column <= threshold
                right_mask = ~left_mask
            case _:
                left_mask = X_column == threshold
                right_mask = ~left_mask

        # Get the child node samples
        left_y = y[left_mask]
        right_y = y[right_mask]

        # If split creates empty node or single-element node, return infinity
        if len(left_y) <= 0 or len(right_y) <= 0:
            return np.inf

        # Calculate MSE for both splits
        left_mean = np.mean(left_y, where=~np.isnan(left_y))
        right_mean = np.mean(right_y, where=~np.isnan(right_y))

        # Handle potential division by zero or nan values
        mse_left = np.mean((left_y - left_mean) ** 2, where=~np.isnan(left_y))
        mse_right = np.mean((right_y - right_mean) ** 2, where=~np.isnan(right_y))

        # Calculate weighted average MSE
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        weighted_mse = (n_left * mse_left + n_right * mse_right) / n_total
        return weighted_mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values for X."""
        return np.array([self._predict_row(row) for row in X])

    def _predict_row(self, row: np.ndarray) -> float:
        """Predict value for a single row."""
        node = self.tree
        while True:
            match node:
                case LeafNode(value=value):
                    return value
                case ParentNode(
                    feature=feature, threshold=threshold, left=left, right=right
                ):
                    if row[feature] <= threshold:
                        node = left
                    else:
                        node = right

    def prune_from_nans(self):
        """Prune the tree from NaN values."""
        self.tree = self._prune_from_nans_recursive(self.tree)

    def _prune_from_nans_recursive(self, node: Node) -> Node:
        """Recursively prune nodes with NaN values, returning the new (potentially modified) node."""
        match node:
            case LeafNode():
                return node
            case ParentNode(feature=feature, threshold=threshold, left=left, right=right):
                # Recursively prune children
                new_left = self._prune_from_nans_recursive(left)
                new_right = self._prune_from_nans_recursive(right)

                # If both children are leaves and one has NaN, merge into a single leaf
                if (
                    isinstance(new_left, LeafNode)
                    and isinstance(new_right, LeafNode)
                ):
                    left_value = new_left.value
                    right_value = new_right.value
                    if np.isnan(left_value) and not np.isnan(right_value):
                        return LeafNode(value=right_value)
                    if np.isnan(right_value) and not np.isnan(left_value):
                        return LeafNode(value=left_value)
                    if np.isnan(left_value) and np.isnan(right_value):
                        return LeafNode(value=np.nan)
                
                # If no pruning needed, return parent node with potentially new children
                return ParentNode(
                    feature=feature,
                    threshold=threshold,
                    left=new_left,
                    right=new_right
                )

    def __str__(self) -> str:
        """Return string representation of the tree."""

        if not self.tree:
            return "Tree not fitted yet"

        def _to_str(node: Node, level: int = 0) -> str:
            indent = "\t" * level

            match node:
                case LeafNode(value=value):
                    return f"{indent}return {value}"
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
