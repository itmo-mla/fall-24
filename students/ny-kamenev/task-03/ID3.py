import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None, left_proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.left_proba = left_proba
        self.num_elems = 0

    def is_leaf(self):
        return self.left is None and self.right is None


class Classifier:
    def __init__(self, max_depth, criteria, prune=False):
        self.root = None
        self.max_depth = max_depth
        self.criteria = criteria
        self.probabilities = {}
        self.max_probability = 0
        self.major = None
        self.prune = prune

    def get_tree(self):
        return self.root

    def fit(self, X, y):
        for label in set(y):
            self.probabilities[label] = len(y.loc[y == label]) / len(y)
        feature_names = X.columns.tolist()
        self.root = self.create_tree(X, y, 0, feature_names)
        self.major = y.value_counts().idxmax()
        if self.prune:
            self.prune_tree(self.root, X, y)

    def predict_instance(self, node, sample):
        if node.is_leaf():
            return node.prediction
        if pd.isna(sample[node.feature]):
            return self.handle_nan(node, sample)
        if sample[node.feature] <= node.threshold:
            return self.predict_instance(node.left, sample)
        else:
            return self.predict_instance(node.right, sample)

    def handle_nan(self, node, sample):
        if node.prediction is not None:
            return node.prediction
        left_prediction = self.predict_instance(node.left, sample)
        right_prediction = self.predict_instance(node.right, sample)
        return left_prediction if node.left_proba > 0.5 else right_prediction

    def prune_tree(self, node, X, y):
        if node.is_leaf():
            return

        if node.left is not None:
            self.prune_tree(node.left, X[X[node.feature] <= node.threshold], y[X[node.feature] <= node.threshold])
        if node.right is not None:
            self.prune_tree(node.right, X[X[node.feature] > node.threshold], y[X[node.feature] > node.threshold])

        if node.left.is_leaf() and node.right.is_leaf():
            left_predictions = y[X[node.feature] <= node.threshold]
            right_predictions = y[X[node.feature] > node.threshold]

            combined_prediction = pd.concat([left_predictions, right_predictions]).mode()[0]

            if combined_prediction == node.left.prediction == node.right.prediction:
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.prediction = combined_prediction

    def predict(self, X):
        predictions = [self.predict_instance(self.root, sample) for _, sample in X.iterrows()]
        return predictions

    @staticmethod
    def entropy(y):
        total_count = len(y)
        label_counts = {}
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        entropy_value = 0.0

        for count in label_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy_value -= probability * np.log2(probability)

        return entropy_value

    @staticmethod
    def donskoy(y):
        total_count = len(y)
        label_counts = {}

        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        donskoy_value = 1.0

        for count in label_counts.values():
            probability = count / total_count
            donskoy_value -= probability ** 2

        return donskoy_value

    def information_gain(self, y, x_column, threshold):
        left_mask = x_column <= threshold
        right_mask = x_column > threshold

        y_left = y[left_mask].dropna()
        y_right = y[right_mask].dropna()
        total = len(y.dropna())

        if len(y_left) == 0 or len(y_right) == 0:
            return 0, 0

        left_count = len(y_left)
        right_count = len(y_right)
        left_probability = left_count / total

        if self.criteria == 'entropy':
            parent_entropy = self.entropy(y)
            child_entropy = (left_count / total) * self.entropy(y_left) + (right_count / total) * self.entropy(y_right)
            return parent_entropy - child_entropy, left_probability
        elif self.criteria == 'donskoy':
            return self.donskoy(y), left_probability
        else:
            raise ValueError("criteria must be either 'entropy' or 'donskoy'")

    def best_split(self, X, y, feature_names):
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_proba = None

        for feature in feature_names:
            thresholds = set(X[feature].dropna())

            for threshold in thresholds:
                gain, left_proba = self.information_gain(y, X[feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_proba = left_proba

        return best_feature, best_threshold, best_left_proba

    def create_tree(self, X, y, depth, feature_names):
        if len(set(y.dropna())) == 1:
            return TreeNode(prediction=y.dropna().iloc[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(prediction=y.mode()[0])

        if X.shape[1] == 0:
            return TreeNode(prediction=y.mode()[0])

        feature, threshold, left_proba = self.best_split(X, y, feature_names)

        if feature is None:
            return TreeNode(prediction=y.mode()[0])

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        left_node = self.create_tree(X[left_mask], y[left_mask], depth + 1, feature_names)
        right_node = self.create_tree(X[right_mask], y[right_mask], depth + 1, feature_names)

        return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node, left_proba=left_proba)


class Regressor(Classifier):

    @staticmethod
    def mean_squared_error(y):
        mean_value = np.mean(y)
        mse = np.mean((y - mean_value) ** 2)
        return mse

    def information_gain(self, y, x_column, threshold):
        left_mask = x_column <= threshold
        right_mask = x_column > threshold

        y_left = y[left_mask].dropna()
        y_right = y[right_mask].dropna()
        total = len(y.dropna())

        if len(y_left) == 0 or len(y_right) == 0:
            return 0, 0

        left_count = len(y_left)
        right_count = len(y_right)
        left_probability = left_count / total

        if self.criteria == 'entropy':
            parent_entropy = self.entropy(y)
            child_entropy = (left_count / total) * self.entropy(y_left) + (right_count / total) * self.entropy(y_right)
            return parent_entropy - child_entropy, left_probability

        elif self.criteria == 'donskoy':
            parent_donskoy = self.donskoy(y)
            child_donskoy = (left_count / total) * self.donskoy(y_left) + (right_count / total) * self.donskoy(y_right)
            return parent_donskoy - child_donskoy, left_probability

        elif self.criteria == 'mse':
            parent_mse = self.mean_squared_error(y)
            child_mse = (left_count / total) * self.mean_squared_error(y_left) + (
                        right_count / total) * self.mean_squared_error(y_right)
            return parent_mse - child_mse, left_probability

        else:
            raise ValueError("criteria must be either 'entropy', 'donskoy', or 'mse'")

    def best_split(self, X, y, feature_names):
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_proba = None

        for feature in feature_names:
            thresholds = set(X[feature].dropna())

            for threshold in thresholds:
                gain, left_proba = self.information_gain(y, X[feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_proba = left_proba

        return best_feature, best_threshold, best_left_proba

    def create_tree(self, X, y, depth, feature_names):
        if len(set(y.dropna())) == 1:
            return TreeNode(prediction=np.mean(y.dropna()))

        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(prediction=np.mean(y))

        if X.shape[1] == 0:
            return TreeNode(prediction=np.mean(y))

        feature, threshold, left_proba = self.best_split(X, y, feature_names)

        if feature is None:
            return TreeNode(prediction=np.mean(y))

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        left_node = self.create_tree(X[left_mask], y[left_mask], depth + 1, feature_names)
        right_node = self.create_tree(X[right_mask], y[right_mask], depth + 1, feature_names)

        return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node, left_proba=left_proba)

    def predict_instance(self, node, sample):
        if node.is_leaf():
            return node.prediction
        if pd.isna(sample[node.feature]):
            return self.handle_nan(node, sample)
        if sample[node.feature] <= node.threshold:
            return self.predict_instance(node.left, sample)
        else:
            return self.predict_instance(node.right, sample)

    def predict(self, X):
        predictions = [self.predict_instance(self.root, sample) for _, sample in X.iterrows()]
        return predictions

    def prune_tree(self, node, X, y):
        if node.is_leaf():
            return

        if node.left is not None:
            self.prune_tree(node.left, X[X[node.feature] <= node.threshold], y[X[node.feature] <= node.threshold])
        if node.right is not None:
            self.prune_tree(node.right, X[X[node.feature] > node.threshold], y[X[node.feature] > node.threshold])

        if node.left.is_leaf() and node.right.is_leaf():
            left_predictions = y[X[node.feature] <= node.threshold]
            right_predictions = y[X[node.feature] > node.threshold]

            combined_prediction = np.mean(pd.concat([left_predictions, right_predictions]))

            current_mse = np.mean((pd.concat([left_predictions, right_predictions]) - combined_prediction) ** 2)
            left_mse = np.mean((left_predictions - node.left.prediction) ** 2)
            right_mse = np.mean((right_predictions - node.right.prediction) ** 2)

            if current_mse <= left_mse + right_mse:
                node.feature = None
                node.threshold = None
                node.left = None
                node.right = None
                node.prediction = combined_prediction