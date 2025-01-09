from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd


class Node:
    def __init__(self, leaf: bool, column: str = None, value: float = None,
                 left: 'Node' = None, right: 'Node' = None, left_prob: float = 0):
        self.leaf = leaf
        self.left = left
        self.right = right
        self.value = value
        self.column = column
        self.left_prob = left_prob

    def print_tree(self):
        return self._print(0)

    def _print(self, depth, is_right=False):
        print('\t' * depth, end='')
        if self.leaf:
            print(f'leaf_{"right" if is_right else "left"}: {self.value}')
            return
        print(f'{self.column} > {self.value}', sep='')
        self.left._print(depth + 1)
        self.right._print(depth + 1, is_right=True)

    def traverse(self, row: pd.Series):
        if self.leaf:
            return self.value
        # Process nan values
        if np.isnan(row[self.column]):
                left_proba = self.left.traverse(row)
                right_proba = self.right.traverse(row)
                return self.left_prob * left_proba + (1 - self.left_prob) * right_proba
        if row[self.column] > self.value:
            return self.right.traverse(row)
        return self.left.traverse(row)

    def post_prune(self, X: pd.DataFrame, y: pd.Series):
        if self.leaf:
            return

        self.left.post_prune(X, y)
        self.right.post_prune(X, y)

        errors = {'left': 0, 'right': 0, 'self': 0}
        for i, row in X.iterrows():
            errors['left'] += self.left.traverse(row) != y.loc[i]
            errors['right'] += self.right.traverse(row) != y.loc[i]
            errors['self'] += self.traverse(row) != y.loc[i]

        minkey = min(errors, key=errors.get)
        if minkey == 'self':
            return

        replacement = self.left if minkey == 'left' else self.right
        self.value = replacement.value
        self.column = replacement.column
        self.left = replacement.left
        self.right = replacement.right
        self.leaf = replacement.leaf
        self.left_prob = replacement.left_prob

class DecisionTree:
    def __init__(self, max_depth, min_samples_split, max_leafs, criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.root = None
        self.criterion = criterion
        self.criterion_func = {
            'entropy': self._calc_entropy,
            'gini': self._calc_gini,
            'mse': self._calc_mse,
            'donskoy': lambda _: -1 # donskoy is handled with _calc_donskoy function
        }.get(self.criterion)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        mask = ~np.any(np.isnan(X), axis=1)
        X, y = X[mask], y[mask]
        self.targets = y.index
        self.root = self._build_tree(X, y, 0, 0, 0)
    def predict(self, X: pd.DataFrame):
        y = {}
        for i, x in X.iterrows():
            y[i] = self.root.traverse(x)
        return pd.Series(y)
    def prune(self, X: pd.DataFrame, y: pd.Series):
        self.root.post_prune(X, y, 0)

    def print_tree(self):
        self.root.print_tree()
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int, left: int, right: int) -> Node:
        if depth == self.max_depth or (self.leafs_cnt + left + 1 >= self.max_leafs and depth > 0) or self.min_samples_split >= len(y):
            self.leafs_cnt += 1
            return Node(leaf=True, value=self._get_leaf_value(y))

        bs_name, bs_val, bs_ig = self._get_best_split(X, y)
        if bs_name is None:
            self.leafs_cnt += 1
            return Node(leaf=True, value=self._get_leaf_value(y))

        left_x, left_y = X[X[bs_name] <= bs_val], y[X[bs_name] <= bs_val]
        right_x, right_y = X[X[bs_name] > bs_val], y[X[bs_name] > bs_val]

        total = len(left_x) + len(right_x)
        left_tree = self._build_tree(left_x, left_y, depth + 1, left + 1, right)
        right_tree = self._build_tree(right_x, right_y, depth + 1, left, right + 1)
        return Node(leaf=False,
                    column=bs_name,
                    value=bs_val,
                    left=left_tree,
                    right=right_tree,
                    left_prob=(len(left_x) / total),
                    )
    def _get_leaf_value(self, values):
        vals, counts = np.unique(values, return_counts=True)
        return vals[np.argmax(counts)]
    def _get_best_split(self, X: pd.DataFrame, y: pd.Series):
        s0 = self.criterion_func(np.array(y))

        if s0 == 0:
            return None, y.iloc[0], 1 # zero entropy, return class value

        name, val, best_ig = None, None, 0

        for col in X.columns:
            current_column = X[col]
            ig, delimiter = self._split_column(s0, current_column, y)
            if ig > best_ig:
                name, val, best_ig = col, delimiter, ig

        return name, val, best_ig
    def _split_column(self, s0, column_data: pd.Series, target_data: pd.Series) -> Tuple[float, float]:
        pd_column_data = column_data.to_numpy()
        pd_target_data = target_data.to_numpy()
        idx_sort = pd_column_data.argsort()

        column_data_sorted = pd_column_data[idx_sort]
        target_data_sorted = pd_target_data[idx_sort]

        delimiters = self._get_delimiters(column_data_sorted)

        best_ig, best_delimiter = 0, None
        for delimiter in delimiters:
            right = target_data_sorted[column_data_sorted > delimiter]
            left = target_data_sorted[column_data_sorted <= delimiter]
            ig = self._calc_donskoy(left, right) if self.criterion == 'donskoy' else self._calc_gain(s0, left, right, target_data)
            if ig > best_ig:
                best_ig, best_delimiter = ig, delimiter

        return best_ig, best_delimiter

    def _calc_gain(self, s0, left, right, target_data):
        s1, k1 = self.criterion_func(left), len(left) / len(target_data)
        s2, k2 = self.criterion_func(right), len(right) / len(target_data)
        return s0 - s1 * k1 - s2 * k2

    def _get_delimiters(self, column_data_sorted: np.ndarray) -> np.array:
        uniq = np.unique(column_data_sorted)
        delimiters = []
        for i in range(1, len(uniq)):
            delimiters.append((uniq[i] + uniq[i - 1]) / 2)
        return np.array(delimiters)

    def _calc_entropy(self, arr: np.array):
        _, counts = np.unique(arr, return_counts=True)
        if len(counts) == 1:
            return 0
        counts = counts.astype(float)
        counts /= len(arr)
        return -sum(counts * np.log2(counts))

    def _calc_gini(self, arr: np.array):
        sm = 0
        for _, count in zip(*np.unique(arr, return_counts=True)):
            sm += count ** 2 / len(arr) ** 2
        return 1 - sm

    def _calc_donskoy(self, left, right):
        l_cnt, r_cnt = Counter(left), Counter(right)
        r_len = len(right)
        return sum(cnt * (r_len - r_cnt.get(v, 0)) for v, cnt in l_cnt.items())

    def _calc_mse(self, arr: np.array):
        ym = np.mean(arr)
        return np.mean((arr - ym) ** 2)


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth, min_samples_split, max_leafs):
        super().__init__(max_depth, min_samples_split, max_leafs, criterion='mse')
    def _calc_gain(self, s0, left, right, target_data):
        s1, k1 = self._calc_mse(left), len(left) / len(target_data)
        s2, k2 = self._calc_mse(right), len(right) / len(target_data)
        return s0 - s1 * k1 - s2 * k2

    def get_leaf_value(self, values):
        return np.mean(values)