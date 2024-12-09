from copy import deepcopy

import numpy as np

from abstract import TreeNode
from criterion import DonskoyCriterion, entropy, multiClassEntropyCriterion


class ClassificationTreeNode(TreeNode):
    def __init__(self, n_features):

        self.n_features = n_features
        self.available_feature_idxs = list(range(self.n_features))

        self.feature_idx = None
        self.beta = None
        self.information_gain = 0
        self.entropy = -1

        self.left_prob = 1

        self.left = None
        self.right = None


    def set_value(self, y):
        self.entropy = entropy(y)
        self.prob = np.array([sum(y==lbl) / len(y) for lbl in range(self.n_features)])


    def fit(self, X, y, criterion="entropy"):
        for feat_idx in self.available_feature_idxs:
            feature = X[:, feat_idx]
            feature_no_nan = feature[~np.isnan(feature)]
            y_no_nan = y[~np.isnan(feature)]
            if criterion == "entropy":
                gain, beta = multiClassEntropyCriterion(feature_no_nan, y_no_nan)
            elif criterion == "donskoy":
                gain, beta = DonskoyCriterion(feature_no_nan, y_no_nan)
            else: raise ValueError(f"Wrong criterion: {criterion}")
            
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
        return self.prob


def build_classification_tree(X, y, criterion="entropy", current_depth=0, max_depth=4) -> ClassificationTreeNode:

    node = ClassificationTreeNode(X.shape[1])
    node.set_value(y)

    if current_depth < max_depth and len(np.unique(y)) > 1:
        (X_0, y_0), (X_1, y_1) = node.fit(X, y, criterion=criterion)

        print(f"Created node with feature: {node.feature_idx}, beta={node.beta}")
        print(f"Сurrent_depth={current_depth}, node entropy: {node.entropy}")
        print(f"Classes in different sets: 0={np.unique(y_0)}, 1={np.unique(y_1)}")
        
        current_depth += 1

        print("\nGo Left")
        node.left = build_classification_tree(X_0, y_0, criterion, current_depth)

        print("\nGo Right")
        node.right = build_classification_tree(X_1, y_1, criterion, current_depth)
    else:
        print(f"Created final node for lbl={np.argmax(node.prob)}, classes={np.unique(y)}")
    return node