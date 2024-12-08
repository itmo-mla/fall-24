import abc
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class TreeNode:
    def __init__(
        self,
        value: Union[float, int, None] = None,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        samples: Optional[int] = None,
        criterion: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
    ):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.samples = samples
        self.criterion = criterion
        self.left = left
        self.right = right
        if not self.is_leaf:
            self.q = left.samples / self.samples
        else:
            self.q = None

    @property
    def label(self):
        if self.is_leaf:
            return self.value
        return f'{self.feature} <= {self.threshold}'

    @property
    def is_leaf(self):
        return not self.left and not self.right

    def print(self, depth=0):
        prefix = '-' * depth
        if self.is_leaf:
            print(f'{prefix} {self.value} ({self.samples})')
        else:
            print(f'{prefix} {self.feature} <= {self.threshold} | '
                  f'criterion: {self.criterion}, samples: {self.samples}')
            if self.left:
                self.left.print(depth + 1)
            if self.right:
                self.right.print(depth + 1)


class BaseDecisionTree(abc.ABC):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.leafs_cnt = 0
        self.bins = bins
        self.tree = None
        self.fi = {}

        self._thresholds = None

    @abc.abstractmethod
    def calculate_gain(self, parent, left, right):
        return NotImplemented

    @abc.abstractmethod
    def get_leaf_value(self, values):
        return NotImplemented

    def prepare_thresholds(self, X: pd.DataFrame):
        if self.bins and self._thresholds is not None:
            return self._thresholds

        thresholds_df = pd.Series()

        for feature in X:
            thresholds = np.unique(X[feature])
            if not self.bins or len(thresholds) - 1 <= self.bins:
                thresholds_df[feature] = np.array([
                    (thresholds[i] + thresholds[i + 1]) / 2
                    for i in range(len(thresholds) - 1)
                ])
            else:
                _, thresholds = np.histogram(X[feature], bins=self.bins)
                thresholds_df[feature] = thresholds[1:-1]

        if self.bins:
            self._thresholds = thresholds_df
        return thresholds_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        mask = ~np.any(np.isnan(X), axis=1)
        X, y = X[mask], y[mask]
        self.fi = {feature: 0 for feature in X.columns}
        _, self.tree = self.build_tree(
            X=X,
            y=y,
            node_indices=X.index,
            levels_left=self.max_depth,
            leaves_left=self.max_leafs,
        )

    def predict(self, X: pd.DataFrame):
        y = np.zeros(X.shape[0])
        i = 0
        for _, x in X.iterrows():
            tree: TreeNode = self.tree
            while not tree.is_leaf:
                if (
                    np.isnan(x[tree.feature]) and tree.q > 0.5
                    or x[tree.feature] <= tree.threshold
                ):
                    tree = tree.left
                else:
                    tree = tree.right
            y[i] = tree.value
            i += 1
        return y

    def build_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        node_indices: np.ndarray,
        levels_left: int,
        leaves_left: int,
    ) -> Tuple[int, TreeNode]:
        split_possible = len(node_indices) >= 1 and y[node_indices].nunique() > 1
        growth_limits = levels_left <= 0 or len(node_indices) < self.min_samples_split or leaves_left <= 1
        if not split_possible or growth_limits:
            self.leafs_cnt += 1
            return 1, TreeNode(
                value=self.get_leaf_value(y[node_indices]),
                criterion=0,
                samples=len(node_indices),
            )

        thresholds = self.prepare_thresholds(X.loc[node_indices])
        best_split = self.get_best_split(X, y, thresholds, node_indices)
        self.fi[best_split['feature']] += len(node_indices) / len(X) * best_split['gain']

        l_leaves, l_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=best_split['left_indices'],
            levels_left=levels_left - 1,
            leaves_left=leaves_left - 1,
        )
        r_leaves, r_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=best_split['right_indices'],
            levels_left=levels_left - 1,
            leaves_left=leaves_left - l_leaves,
        )

        tree = TreeNode(
            feature=best_split['feature'],
            threshold=best_split['threshold'],
            samples=len(node_indices),
            criterion=best_split['gain'],
            left=l_tree,
            right=r_tree,
        )

        return l_leaves + r_leaves, tree

    def print_tree(self):
        self.tree.print()

    def split_data(self, X, feature, threshold, node_indices=None):
        if node_indices is None:
            node_indices = X.index

        left_indices = node_indices[X.loc[node_indices, feature] <= threshold]
        right_indices = node_indices[X.loc[node_indices, feature] > threshold]
        return np.array(left_indices), np.array(right_indices)

    def get_best_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        thresholds: pd.Series,
        node_indices=None,
    ):
        if node_indices is None:
            node_indices = X.index

        best_split = {
            'feature': None,
            'gain': -1,
            'threshold': None,
            'left_indices': None,
            'right_indices': None,
        }

        for feature in X.columns:
            for threshold in thresholds[feature]:
                left_indices, right_indices = self.split_data(X, feature, threshold, node_indices)
                if not len(left_indices) or not len(right_indices):
                    continue

                cur_gain = self.calculate_gain(
                    parent=y[node_indices],
                    left=y[left_indices],
                    right=y[right_indices],
                )

                if cur_gain > best_split['gain']:
                    best_split['feature'] = feature
                    best_split['gain'] = cur_gain
                    best_split['threshold'] = threshold
                    best_split['left_indices'] = left_indices
                    best_split['right_indices'] = right_indices

        return best_split

    @property
    def feature_importances_(self):
        return np.array(list(self.fi.values()))

    def __str__(self):
        return (
            f'{self.__class__.__name__} class: '
            f'max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, '
            f'max_leafs={self.max_leafs}, '
            f'bins={self.bins}'
        )
