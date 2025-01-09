import numpy as np

from base_decision_tree import BaseDecisionTree


def mse(y):
    ym = np.mean(y)
    return np.mean((y - ym) ** 2)


class DecisionTreeRegression(BaseDecisionTree):
    def calculate_gain(self, parent, left, right):
        parent_mse, left_mse, right_mse = map(mse, (parent, left, right))
        left_weight, right_weight = len(left) / len(parent), len(right) / len(parent)
        return parent_mse - (left_weight * left_mse + right_weight * right_mse)

    def get_leaf_value(self, values):
        return np.mean(values)
