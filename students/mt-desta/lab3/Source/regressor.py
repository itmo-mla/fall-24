from copy import deepcopy

import numpy as np


def uncertainty_measure(Y):
    """Calculate mean squared error for a set of values."""
    if len(Y) == 0:
        return 0
    return np.mean((Y - np.mean(Y)) ** 2)


def MSECriterion(X, y):
    """Calculate MSE-based split criterion for regression tree.
    
    Args:
        X: Feature values
        y: Target values
    Returns:
        (min_mse, best_threshold): Tuple of minimum MSE and best threshold
    """
    if len(X) < 2:  # Need at least 2 samples to split
        return float('inf'), None
    
    min_mse = float('inf')
    best_threshold = None
    
    # Use percentile points as candidate splits for better distribution
    percentiles = np.percentile(X, np.linspace(1, 99, 50))
    unique_values = np.unique(percentiles)
    
    parent_mse = uncertainty_measure(y)
    
    for threshold in unique_values:
        left_mask = X <= threshold
        right_mask = ~left_mask
        
        # Skip if split would result in too small nodes
        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            continue
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        # Calculate weighted MSE
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)
        
        left_mse = uncertainty_measure(y_left)
        right_mse = uncertainty_measure(y_right)
        
        weighted_mse = (n_left/n_total) * left_mse + (n_right/n_total) * right_mse
        
        if weighted_mse < min_mse:
            min_mse = weighted_mse
            best_threshold = threshold
    
    improvement = parent_mse - min_mse
    # Only return split if it improves MSE significantly
    if improvement <= 1e-7:
        return float('inf'), None
        
    return min_mse, best_threshold


class TreeNode:
    """Base class for tree nodes"""
    def __init__(self):
        self.feature_idx = None  # Index of the feature used for splitting
        self.beta = None        # Threshold value for the split
        self.left = None        # Left child node
        self.right = None       # Right child node
        self.prob = None        # Probability distribution for classification
        
    def fit(self, X, y):
        """Train the node on the given data"""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X_sample):
        """Make a prediction for the given sample"""
        raise NotImplementedError("Subclasses must implement predict()")
        
    def set_value(self, y):
        """Set the node's value based on the target values"""
        raise NotImplementedError("Subclasses must implement set_value()")


class RegressionTreeNode(TreeNode):
    def __init__(self, n_features):
        self.n_features = n_features
        self.available_feature_idxs = list(range(self.n_features))
        self.feature_idx = None
        self.beta = None
        self.mse = float('inf')
        self.left_prob = 0.5
        self.left = None
        self.right = None
        self.value = None
        self.n_samples = 0

    def set_value(self, y):
        """Set node's value to mean of target values."""
        self.value = np.mean(y)
        self.n_samples = len(y)
        self.mse = uncertainty_measure(y)

    def fit(self, X, y):
        """Find best split for node."""
        if len(np.unique(y)) < 2:  # All values are the same
            return (X, y), (np.array([]), np.array([]))
            
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        best_splits = None
        
        for feat_idx in self.available_feature_idxs:
            feature = X[:, feat_idx]
            valid_mask = ~np.isnan(feature)
            if np.sum(valid_mask) < 2:  # Need at least 2 valid values
                continue
                
            feature_no_nan = feature[valid_mask]
            y_no_nan = y[valid_mask]
            
            mse, threshold = MSECriterion(feature_no_nan, y_no_nan)
            
            if mse < best_mse and threshold is not None:
                mask = ~np.isnan(X[:, feat_idx])
                X_valid = X[mask]
                y_valid = y[mask]
                
                left_mask = X_valid[:, feat_idx] <= threshold
                X_0 = X_valid[left_mask]
                y_0 = y_valid[left_mask]
                X_1 = X_valid[~left_mask]
                y_1 = y_valid[~left_mask]
                
                # Verify split is valid
                if len(y_0) >= 2 and len(y_1) >= 2:
                    best_mse = mse
                    best_feature = feat_idx
                    best_threshold = threshold
                    best_splits = ((X_0, y_0), (X_1, y_1))
        
        if best_threshold is None:  # No valid split found
            return (X, y), (np.array([]), np.array([]))
            
        self.feature_idx = best_feature
        self.beta = best_threshold
        self.mse = best_mse
        
        (X_0, y_0), (X_1, y_1) = best_splits
        total_samples = len(y_0) + len(y_1)
        self.left_prob = len(y_0) / total_samples if total_samples > 0 else 0.5
        
        return best_splits

    def predict(self, X_sample):
        """Make a prediction for a single sample.
        
        Args:
            X_sample: Single sample to predict
        Returns:
            float: Predicted value
        """
        # If this is a leaf node or no valid split was found
        if self.left is None or self.right is None:
            return self.value
            
        # Get the feature value for the split
        feat_value = X_sample[self.feature_idx]
        
        # Handle missing values
        if np.isnan(feat_value):
            # Use weighted average of child predictions
            left_pred = self.left.predict(X_sample)
            right_pred = self.right.predict(X_sample)
            return self.left_prob * left_pred + (1 - self.left_prob) * right_pred
            
        # Normal prediction based on split
        if feat_value <= self.beta:
            return self.left.predict(X_sample)
        else:
            return self.right.predict(X_sample)


def build_regression_tree(X, y, current_depth=0, max_depth=4) -> RegressionTreeNode:
    """Build a regression tree recursively.
    
    Args:
        X: Feature matrix
        y: Target values
        current_depth: Current depth in the tree
        max_depth: Maximum allowed depth
    Returns:
        RegressionTreeNode: Root node of the tree
    """
    node = RegressionTreeNode(X.shape[1])
    node.set_value(y)
    
    # Stop conditions
    if (len(y) < 5 or  # Too few samples
        current_depth >= max_depth or  # Max depth reached
        len(np.unique(y)) < 2):  # All values same
        return node
    
    (X_0, y_0), (X_1, y_1) = node.fit(X, y)
    
    # Only split if we got valid splits
    if len(y_0) >= 2 and len(y_1) >= 2:
        current_depth += 1
        node.left = build_regression_tree(X_0, y_0, current_depth, max_depth)
        node.right = build_regression_tree(X_1, y_1, current_depth, max_depth)
    
    return node

def prune_regression_tree(node: RegressionTreeNode, X, y):
    """Prune a regression tree using validation data.
    
    Args:
        node: The tree node to prune
        X: Validation features
        y: Validation labels
    Returns:
        The pruned node
    """
    if node is None or node.beta is None:
        return node

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