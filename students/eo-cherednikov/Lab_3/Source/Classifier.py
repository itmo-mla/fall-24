import numpy as np

class TreeClassifier:
    def __init__(self, criterion: str = 'entropy', max_depth: int = None, min_samples_split: int = 2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.tree = None

        self.class_probabilities = None
        self.majority_class = None


    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:

        valid_indices = ~np.isnan(features).any(axis=1)
        features = features[valid_indices]
        labels = labels[valid_indices]

        unique_classes, counts = np.unique(labels, return_counts=True)
        self.majority_class = unique_classes[np.argmax(counts)]
        self.class_probabilities = counts / len(labels)

        self.tree = self._id3_classification(
            features,
            labels,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )


    def _id3_classification(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            depth: int,
            max_depth: int,
            min_samples_split: int
    ):
        if max_depth is not None and depth >= max_depth:
            return self._make_leaf(labels)

        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return self._make_leaf(labels)

        if len(labels) < min_samples_split:
            return self._make_leaf(labels)

        best_feature = None
        best_threshold = None
        best_info_gain = -np.inf

        n_samples, n_features = features.shape
        for feature_index in range(n_features):
            thresholds = np.unique(features[:, feature_index])

            for threshold in thresholds:
                mask = features[:, feature_index] <= threshold

                left_count = np.sum(mask)
                right_count = n_samples - left_count
                if left_count == 0 or right_count == 0:
                    continue

                info_gain = (
                    self._calculate_information_gain(labels, mask)
                    if self.criterion == 'entropy'
                    else self._d_criterion(labels, mask)
                )

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return self._make_leaf(labels)

        mask = features[:, best_feature] <= best_threshold
        left_count = np.sum(mask)
        right_count = len(labels) - left_count

        if left_count == 0 or right_count == 0:
            return self._make_leaf(labels)

        left_subtree = self._id3_classification(
            features[mask],
            labels[mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        right_subtree = self._id3_classification(
            features[~mask],
            labels[~mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        left_probability = left_count / len(labels)

        return (best_feature, best_threshold, left_subtree, right_subtree, left_probability)


    def _make_leaf(self, labels: np.ndarray):
        if len(labels) == 0:
            return (self.majority_class, None, None, None, None)
        counts = np.bincount(labels)
        probabilities = counts / len(labels)
        return (counts.argmax(), None, None, None, probabilities)


    def _calculate_information_gain(self, labels: np.ndarray, mask: np.ndarray) -> float:
        total_entropy = self._entropy(labels)
        left_entropy = self._entropy(labels[mask])
        right_entropy = self._entropy(labels[~mask])

        left_ratio = np.sum(mask) / len(labels)
        right_ratio = 1.0 - left_ratio
        weighted_entropy = left_ratio * left_entropy + right_ratio * right_entropy
        return total_entropy - weighted_entropy


    def _d_criterion(self, labels: np.ndarray, mask: np.ndarray) -> float:
        return self._calculate_information_gain(labels, mask)


    def _entropy(self, labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))


    def predict(self, samples: np.ndarray) -> np.ndarray:
        predictions = self._predict_batch(samples, self.tree)
        return predictions


    def _predict_batch(self, samples: np.ndarray, tree) -> np.ndarray:
        return np.array([self._predict(sample, tree) for sample in samples])


    def _predict(self, sample: np.ndarray, tree) -> int:
        feature, threshold, left_subtree, right_subtree, _ = tree

        if threshold is None:
            return feature

        if np.isnan(sample).any():
            return self.majority_class

        if sample[feature] <= threshold:
            return self._predict(sample, left_subtree)
        else:
            return self._predict(sample, right_subtree)


    def prune(self, validation_features: np.ndarray, validation_labels: np.ndarray) -> None:
        self.tree = self._prune_tree(self.tree, validation_features, validation_labels)


    def _prune_tree(self, tree, val_features: np.ndarray, val_labels: np.ndarray):
        feature, threshold, left_subtree, right_subtree, _ = tree

        if threshold is None:
            return tree

        mask = val_features[:, feature] <= threshold
        left_val_features = val_features[mask]
        left_val_labels = val_labels[mask]
        right_val_features = val_features[~mask]
        right_val_labels = val_labels[~mask]

        if left_subtree is not None:
            left_pruned = self._prune_tree(left_subtree, left_val_features, left_val_labels)
        else:
            left_pruned = None
        if right_subtree is not None:
            right_pruned = self._prune_tree(right_subtree, right_val_features, right_val_labels)
        else:
            right_pruned = None

        pruned_tree = (feature, threshold, left_pruned, right_pruned, _)

        original_accuracy = self._calculate_accuracy(tree, val_features, val_labels)
        pruned_accuracy = self._calculate_accuracy(pruned_tree, val_features, val_labels)

        if pruned_accuracy >= original_accuracy:
            return pruned_tree
        else:
            class_label = self._get_majority_class(val_labels)
            return (class_label, None, None, None, None)


    def _get_majority_class(self, labels: np.ndarray) -> int:
        if len(labels) == 0:
            return self.majority_class
        unique_classes, counts = np.unique(labels, return_counts=True)
        return unique_classes[np.argmax(counts)]


    def _calculate_accuracy(self, tree, val_features: np.ndarray, val_labels: np.ndarray) -> float:
        predictions = self._predict_batch(val_features, tree)
        return np.mean(predictions == val_labels)
