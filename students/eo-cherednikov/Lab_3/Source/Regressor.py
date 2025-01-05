import numpy as np


class TreeRegressor:
    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

        self.class_distribution = None
        self.majority_value = None


    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        valid_idx = ~np.isnan(features).any(axis=1)
        features = features[valid_idx]
        targets = targets[valid_idx]

        unique_vals, counts = np.unique(targets, return_counts=True)
        self.majority_value = unique_vals[np.argmax(counts)]

        self.class_distribution = counts / len(targets)

        self.root = self._build_tree(features, targets, depth=0)


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if (self.max_depth is not None) and (depth >= self.max_depth):
            return self._make_leaf(y)

        if (X.shape[0] < self.min_samples_split) or (len(np.unique(y)) == 1):
            return self._make_leaf(y)

        best_feat, best_thresh, best_gain = None, None, -np.inf

        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            unique_vals = np.unique(X[:, feature_idx])
            for threshold in unique_vals:
                mask = X[:, feature_idx] <= threshold
                gain = self._rmse_reduction(y, mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_idx
                    best_thresh = threshold

        if best_gain == -np.inf:
            return self._make_leaf(y)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        left_prob = len(y[left_mask]) / len(y)

        return (best_feat, best_thresh, left_subtree, right_subtree, left_prob)


    def _make_leaf(self, y: np.ndarray):
        leaf_val = np.mean(y)
        prob_info = (self.class_distribution[0]
                     if self.class_distribution is not None
                     else 0.0)
        return (leaf_val, None, None, leaf_val, prob_info)


    def _rmse(self, values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        return np.sqrt(np.mean((values - np.mean(values)) ** 2))


    def _rmse_reduction(self, y: np.ndarray, mask: np.ndarray) -> float:
        total_rmse = self._rmse(y)
        left_rmse = self._rmse(y[mask])
        right_rmse = self._rmse(y[~mask])
        w_left = np.sum(mask) / len(y)
        w_right = 1.0 - w_left
        weighted_rmse = w_left * left_rmse + w_right * right_rmse
        return total_rmse - weighted_rmse


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(sample, self.root) for sample in X])


    def _predict_single(self, sample: np.ndarray, node) -> float:
        feature, threshold, left_subtree, right_subtree, _ = node

        if threshold is None:
            return feature

        if np.isnan(sample[feature]):
            return self.majority_value

        if sample[feature] <= threshold:
            return self._predict_single(sample, left_subtree)
        else:
            return self._predict_single(sample, right_subtree)


    def prune(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self.root = self._prune_tree(self.root, X_val, y_val)


    def _prune_tree(self, node, X_val: np.ndarray, y_val: np.ndarray):
        feature, threshold, left_subtree, right_subtree, prob_info = node

        if threshold is None:
            return node

        left_subtree = self._prune_tree(left_subtree, X_val, y_val)
        right_subtree = self._prune_tree(right_subtree, X_val, y_val)
        pruned_node = (feature, threshold, left_subtree, right_subtree, prob_info)

        current_error = self._evaluate(node, X_val, y_val)
        pruned_error = self._evaluate(pruned_node, X_val, y_val)

        if pruned_error <= current_error:
            return pruned_node
        else:
            # Иначе делаем лист из текущего узла
            leaf_val = np.mean(y_val)
            return (leaf_val, None, None, leaf_val, prob_info)


    def _evaluate(self, node, X: np.ndarray, y: np.ndarray) -> float:
        preds = self._predict_batch(X, node)
        return np.mean((y - preds) ** 2)


    def _predict_batch(self, X: np.ndarray, node) -> np.ndarray:
        return np.array([self._predict_single(sample, node) for sample in X])
