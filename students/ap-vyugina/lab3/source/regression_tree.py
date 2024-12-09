from copy import deepcopy

import numpy as np

from abstract import TreeNode
from criterion import MSECriterion, uncertainty_measure


class RegressionTreeNode(TreeNode):
    def __init__(self, n_features):

        self.n_features = n_features
        self.available_feature_idxs = list(range(self.n_features))

        self.feature_idx = None
        self.beta = None
        self.information_gain = 0
        self.uncertainty = -1

        self.left_prob = 1
        self.left = None
        self.right = None


    def set_value(self, y):
        self.uncertainty = uncertainty_measure(y)
        self.value = np.mean(y)


    def fit(self, X, y):
        for feat_idx in self.available_feature_idxs:
            feature = X[:, feat_idx]
            feature_no_nan = feature[~np.isnan(feature)]
            y_no_nan = y[~np.isnan(feature)]

            gain, beta = MSECriterion(feature_no_nan, y_no_nan)
            
            if gain > self.information_gain:
                self.information_gain = gain
                self.beta = beta
                self.feature_idx = feat_idx
        
        X_0 = X[(X[:, self.feature_idx] <= self.beta) & ~np.isnan(feature)]
        y_0 = y[(X[:, self.feature_idx] <= self.beta) & ~np.isnan(feature)]

        X_1 = X[(X[:, self.feature_idx] > self.beta) & ~np.isnan(feature)]
        y_1 = y[(X[:, self.feature_idx] > self.beta) & ~np.isnan(feature)]

        self.left_prob = len(X_0) / (len(X_0) + len(X_1))
        return (X_0, y_0), (X_1, y_1)
    

    def predict(self, X_sample):
        if self.beta is not None:
            feat_value = X_sample[self.feature_idx]
            if np.isnan(feat_value):
                # если текущая нода - лист
                if self.left is None or self.right is None:
                    pass
                # если текущая нода - ветка
                else:
                    return self.left_prob * self.left.predict(X_sample) + (1 - self.left_prob) * self.right.predict(X_sample)
            else:
                if feat_value <= self.beta:
                    return self.left.predict(X_sample)
                else:
                    return self.right.predict(X_sample)
        return self.value


def build_regression_tree(X, y, current_depth=0, max_depth=4) -> RegressionTreeNode:

    node = RegressionTreeNode(X.shape[1])
    node.set_value(y)

    if current_depth < max_depth-1 and len(np.unique(y)) > 1:
        (X_0, y_0), (X_1, y_1) = node.fit(X, y)

        print(f"Created node with feature: {node.feature_idx}, beta={node.beta}")
        print(f"Сurrent_depth={current_depth}, node uncertainty: {node.uncertainty:.3f}")
        print(f"Dispersion in different sets: 0={uncertainty_measure(y_0):.3f}, 1={uncertainty_measure(y_1):.3f}")
        
        current_depth += 1
        
        print("\nGo Left")
        node.left = build_regression_tree(X_0, y_0, current_depth, max_depth=max_depth)

        print("\nGo Right")
        node.right = build_regression_tree(X_1, y_1, current_depth, max_depth=max_depth)
    else:
        print(f"Created final node for value={node.value:.2f}")
    return node