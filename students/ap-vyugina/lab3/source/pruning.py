import numpy as np

from classification_tree import ClassificationTreeNode
from regression_tree import RegressionTreeNode


### Classification
def prune_classification_tree(node: ClassificationTreeNode, X, y):

    def compute_errors(node: ClassificationTreeNode, X, y):
        y_curr, y_left, y_right = [], [], []
        for x in X:
            x = x.reshape(-1, 1)
            y_curr += [np.argmax(node.predict(x))]
            y_left += [np.argmax(node.left.predict(x))]
            y_right += [np.argmax(node.right.predict(x))]

        y = y.flatten()
        err_curr = sum(y_curr != y) / len(y)
        err_left = sum(y_left != y) / len(y) 
        err_right = sum(y_right != y) / len(y) 

        most_freq = np.argmax(np.bincount(y))
        err_base = sum(y != most_freq) / len(y)
        return (err_curr, err_left, err_right, err_base)

    # node, left, right, major_cls
    errors = compute_errors(node, X, y)
    min_err_idx = np.argmin(errors)

    print(errors, min_err_idx)
    if min_err_idx == 0: # keep this node
        pass
    elif min_err_idx == 1: # replace with left
        node = node.left
    elif min_err_idx == 2: # replace with right
        node = node.right
    else: # create a leaf wit max. frequent value
        node = ClassificationTreeNode(X.shape[1])
        node.set_value(np.argmax(np.bincount(y)))
    
    if min_err_idx != 3: # if current node didn't become a leaf
        X_0 = X[(X[:, node.feature_idx] <= node.beta)]
        y_0 = y[(X[:, node.feature_idx] <= node.beta)]

        X_1 = X[(X[:, node.feature_idx] > node.beta)]
        y_1 = y[(X[:, node.feature_idx] > node.beta)]

    if node.beta is not None:
        if node.left.beta is not None:
            node.left = prune_classification_tree(node.left, X_0, y_0)
        if node.right.beta is not None:
            node.right = prune_classification_tree(node.right, X_1, y_1)

    return node


### Regression
def prune_regression_tree(node: RegressionTreeNode, X, y):

    def compute_errors(node: RegressionTreeNode, X, y):
        y_curr, y_left, y_right = [], [], []
        for x in X:
            x = x.reshape(-1, 1)
            y_curr += [node.predict(x)]
            y_left += [node.left.predict(x)]
            y_right += [node.right.predict(x)]
        
        y = y.flatten()
        mean_y = np.mean(y)
        
        err_curr = np.sum((y_curr - y)**2) / np.sum((mean_y - y)**2)
        err_left = np.sum((y_left - y)**2) / np.sum((mean_y - y)**2)
        err_right = np.sum((y_right - y)**2) / np.sum((mean_y - y)**2)
        err_base = np.sum((mean_y - y)**2) / np.sum((mean_y - y)**2)

        return err_curr, err_left, err_right, err_base

    # node, left, right, major_cls
    errors = compute_errors(node, X, y)
    min_err_idx = np.argmin(errors)

    print(errors, min_err_idx)
    if min_err_idx == 0: # keep this node
        pass
    elif min_err_idx == 1: # replace with left
        node = node.left
    elif min_err_idx == 2: # replace with right
        node = node.right
    else: # create a leaf wit max. frequent value
        node = RegressionTreeNode(X.shape[1])
        node.set_value(y)
    
    if min_err_idx != 3: # if current node didn't become a leaf
        X_0 = X[(X[:, node.feature_idx] <= node.beta)]
        y_0 = y[(X[:, node.feature_idx] <= node.beta)]

        X_1 = X[(X[:, node.feature_idx] > node.beta)]
        y_1 = y[(X[:, node.feature_idx] > node.beta)]

    if node.beta is not None:
        if node.left.beta is not None:
            node.left = prune_regression_tree(node.left, X_0, y_0)
        if node.right.beta is not None:
            node.right = prune_regression_tree(node.right, X_1, y_1)

    return node