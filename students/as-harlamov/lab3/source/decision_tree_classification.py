from collections import Counter
from typing import Literal, Optional

import numpy as np

from base_decision_tree import BaseDecisionTree


def entropy(y):
    ans = 0
    n = len(y)
    for _, count in zip(*np.unique(y, return_counts=True)):
        p = count / n
        ans += -p * np.log2(p)
    return ans


def gini(y):
    sm = 0
    for _, count in zip(*np.unique(y, return_counts=True)):
        sm += count ** 2 / len(y) ** 2
    return 1 - sm


def donskoy(left, right):
    l_cnt, r_cnt = Counter(left), Counter(right)
    r_len = len(right)
    return sum(cnt * (r_len - r_cnt.get(v, 0)) for v, cnt in l_cnt.items())


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
        criterion: Literal['entropy', 'gini', 'donskoy'] = 'gini',
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_leafs=max_leafs,
            bins=bins,
        )
        self.criterion = criterion
        self.gain_func = {'entropy': entropy, 'gini': gini}.get(self.criterion)

    def get_leaf_value(self, values):
        vals, counts = np.unique(values, return_counts=True)
        return vals[np.argmax(counts)]

    def calculate_gain(self, parent, left, right):
        if self.criterion == 'donskoy':
            return donskoy(left, right)

        parent_ambiguity, left_ambiguity, right_ambiguity = map(self.gain_func, (parent, left, right))
        left_weight, right_weight = len(left) / len(parent), len(right) / len(parent)
        weighted_ambiguity = left_weight * left_ambiguity + right_weight * right_ambiguity
        return parent_ambiguity - weighted_ambiguity
