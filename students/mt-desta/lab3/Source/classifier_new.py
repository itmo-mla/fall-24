from copy import deepcopy

import numpy as np

### Classification

import numpy as np

def entropy(probabilities):
    """
    Compute entropy for a probability value or array.
    Handles the case where probabilities contain zeros gracefully.
    """
    probabilities = np.clip(probabilities, 1e-10, 1)  # Avoid log(0) by clipping probabilities
    return -probabilities * np.log2(probabilities)

def multiClassEntropyCriterion(feature_values, class_labels):
    """
    Calculate the best split threshold for a single feature using
    multiclass entropy as the criterion.

    Parameters:
    feature_values: 1D array of feature values.
    class_labels: 1D array of class labels.

    Returns:
    max_information_gain: The maximum information gain achieved.
    best_split_threshold: The threshold value corresponding to the maximum information gain.
    """
    max_information_gain = -1
    best_split_threshold = None

    total_samples = len(feature_values)
    unique_classes, class_counts_parent = np.unique(class_labels, return_counts=True)

    sorted_thresholds = np.unique(feature_values)[:-1]

    for threshold in sorted_thresholds:
        # Split based on the threshold
        right_child_mask = feature_values > threshold
        right_child_count = np.sum(right_child_mask)
        
        if right_child_count == 0 or right_child_count == total_samples:
            continue  # Skip invalid splits

        class_counts_right_child = np.array([np.sum((class_labels == class_label) & right_child_mask) for class_label in unique_classes])
        class_counts_left_child = class_counts_parent - class_counts_right_child

        # Compute entropy values
        parent_entropy = np.sum(entropy(class_counts_parent / total_samples))
        right_child_entropy = np.sum(entropy(class_counts_right_child / right_child_count))
        left_child_entropy = np.sum(entropy(class_counts_left_child / (total_samples - right_child_count)))

        # Compute information gain
        information_gain = (parent_entropy -
                            (right_child_count / total_samples) * right_child_entropy -
                            ((total_samples - right_child_count) / total_samples) * left_child_entropy)

        # Update best split if information gain is improved
        if information_gain > max_information_gain:
            max_information_gain = round(information_gain, 5)
            best_split_threshold = threshold

    return max_information_gain, best_split_threshold



def DonskoyCriterion(feature_values, class_labels):
    max_information_gain = -1
    best_split_threshold = None

    unique_thresholds = sorted(np.unique(feature_values))[:-1]

    for threshold in unique_thresholds:
        # Split based on the threshold
        split_mask = feature_values > threshold
        split_counts = np.sum(split_mask)

        if split_counts == 0 or split_counts == len(feature_values):
            continue  # Skip invalid splits

        # Calculate the information gain
        information_gain = np.sum((split_mask[:, None] != split_mask) & (class_labels[:, None] != class_labels))

        # Update best split if information gain is improved
        if information_gain > max_information_gain:
            max_information_gain = round(information_gain, 5)
            best_split_threshold = threshold

    return max_information_gain, best_split_threshold



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


class ClassificationTreeNode(TreeNode):
    def __init__(self, n_features):
        super().__init__()  # Call parent class constructor
        self.n_features = n_features
        self.available_feature_idxs = list(range(self.n_features))
        self.information_gain = 0
        self.entropy = -1
        self.left_prob = 1


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

        current_depth += 1
        node.left = build_classification_tree(X_0, y_0, criterion, current_depth)
        node.right = build_classification_tree(X_1, y_1, criterion, current_depth)
    
    return node


def prune_classification_tree(node: ClassificationTreeNode, X, y, n_classes):
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

        most_freq = np.argmax(np.bincount(y, minlength=n_classes))
        err_base = sum(y != most_freq) / len(y)
        return (err_curr, err_left, err_right, err_base)

    # node, left, right, major_cls
    errors = compute_errors(node, X, y)
    min_err_idx = np.argmin(errors)

    if min_err_idx == 0: # keep this node
        pass
    elif min_err_idx == 1: # replace with left
        node = node.left
    elif min_err_idx == 2: # replace with right
        node = node.right
    else: # create a leaf with most frequent class
        node = ClassificationTreeNode(n_classes)
        most_freq = np.argmax(np.bincount(y, minlength=n_classes))
        node.set_value(np.array([most_freq] * len(y)))
    
    if min_err_idx != 3: # if current node didn't become a leaf
        X_0 = X[(X[:, node.feature_idx] <= node.beta)]
        y_0 = y[(X[:, node.feature_idx] <= node.beta)]

        X_1 = X[(X[:, node.feature_idx] > node.beta)]
        y_1 = y[(X[:, node.feature_idx] > node.beta)]

    if node.beta is not None:
        if node.left.beta is not None:
            node.left = prune_classification_tree(node.left, X_0, y_0, n_classes)
        if node.right.beta is not None:
            node.right = prune_classification_tree(node.right, X_1, y_1, n_classes)

    return node