import numpy as np
from read import read_hero
from sklearn.model_selection import train_test_split
from copy import deepcopy

class DecisionTreeClassification:
    def __init__(self, method='entropy'):
        self.method = method
        self.tree = None
        self.probabilities = None

        self.__max_probabilitiesbilities = -1

    def fit(self, X, y):
        not_nan_index = ~np.isnan(X).any(axis=1)
        X = X[not_nan_index]
        y = y[not_nan_index]

        _, counts = np.unique(y, return_counts=True)
        self.majority_class = np.argmax(counts)
        self.probabilities = counts / len(y)
        
        self.tree = self._id3_classification(X, y)

    def __prune_all_tree(self, tree, X_val, y_val):
        feature_index, threshold, left_subtree, right_subtree, left_probabilities = tree

        if threshold is None:
            return tree
        
        left_pruned_tree = self.__prune_all_tree(left_subtree, X_val, y_val)
        right_pruned_tree = self.__prune_all_tree(right_subtree, X_val, y_val)
        
        pruned_tree = (feature_index, threshold, left_pruned_tree, right_pruned_tree, left_probabilities)

        original_accuracy = self.__check_accuracy(tree, X_val, y_val)
        pruned_accuracy = self.__check_accuracy(pruned_tree, X_val, y_val)

        if pruned_accuracy >= original_accuracy:
            return pruned_tree
        else:
            return (self.majority_class, None, None, None, left_probabilities)
        
    def __check_accuracy(self, tree, X_val, y_val):
        predictions = self._predict(X_val, tree)
        return np.mean(predictions == y_val)
    
    def pruning(self, X, y):
        self.tree = self.__prune_all_tree(self.tree, X, y)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilitiesbilities = counts / len(y)
        return -np.sum(probabilitiesbilities * np.log2(probabilitiesbilities))

    def _d_criterion(self, y, mask):
        if len(y) == 0:
            return 0
        
        y_left = y[mask]
        y_right = y[~mask]
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        differences = np.sum(y_left[:, np.newaxis] != y_right[np.newaxis, :], axis=0)
        entropy = np.sum(differences)
        return entropy

    def _calculate_information_gain(self, y, mask):
        total_entropy = self._entropy(y)
        left_entropy = self._entropy(y[mask])
        right_entropy = self._entropy(y[~mask])
        
        weighted_entropy = (len(y[mask]) / len(y)) * left_entropy + (len(y[~mask]) / len(y)) * right_entropy
        return total_entropy - weighted_entropy

    def _id3_classification(self, X, y):
        if len(np.unique(y)) == 1:
            biny = np.bincount(y)
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return biny.argmax(), None, None, None, probabilities[0]
        
        notnon_index = ~np.isnan(X).any(axis=1)
        X = X[notnon_index]
        y = y[notnon_index]
        
        best_feature = None
        best_threshold = None
        best_info_gain = -np.inf
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                mask = X[:, feature_index] <= threshold
                if self.method == 'entropy':
                    info_gain = self._calculate_information_gain(y, mask)
                elif self.method == 'donskoy':
                    info_gain = self._d_criterion(y, mask)
                
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        if best_feature is None:
            biny = np.bincount(y)
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return biny.argmax(), None, None, None, probabilities
        
        mask = X[:, best_feature] <= best_threshold
        left_subtree = self._id3_classification(X[mask], y[mask])
        right_subtree = self._id3_classification(X[~mask], y[~mask])
        left_probabilities = len(y[mask]) / len(y)
        
        return (best_feature, best_threshold, left_subtree, right_subtree, left_probabilities)

    def _predict_non(self, sample, tree, current_probabilitiesbilities):
        feature_index, threshold, left_subtree, _, left_probabilities = tree

        if threshold is None:
            if self.__max_probabilitiesbilities < current_probabilitiesbilities * self.probabilities[feature_index]:
                self.__max_probabilitiesbilities = current_probabilitiesbilities * self.probabilities[feature_index]
                return feature_index
            return self.majority_class
        else:
            if current_probabilitiesbilities != 0:
                l = self._predict_non(sample, left_subtree, left_probabilities * current_probabilitiesbilities)
                r = self._predict_non(sample, left_subtree, (1 - left_probabilities) * current_probabilitiesbilities)
            else:
                l = self._predict_non(sample, left_subtree, left_probabilities)
                r = self._predict_non(sample, left_subtree, (1 - left_probabilities))
            
            return max(l, r)

    def _predict(self, sample, tree):        
        feature_index, threshold, left_subtree, right_subtree, _ = tree

        if threshold is None:
            return feature_index
        
        if not np.isnan(sample).any():
            if sample[feature_index] <= threshold:
                return self._predict(sample, left_subtree)
            else:
                return self._predict(sample, right_subtree)
        else:
            return self._predict_non(sample, self.tree, 0)
            
    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

if __name__ == "__main__":
    X, y = read_hero('dataset/superheroes_data.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    decision_tree = DecisionTreeClassification(method='entropy')
    decision_tree.fit(X_train, y_train)
    pred_y = decision_tree.predict(X_val)
    print("Predict value:", pred_y)
    print("True  value:", y_val)

    pruned_tree = deepcopy(decision_tree)
    pruned_tree.pruning(X_val, y_val)
    pred_y = pruned_tree.predict(X_val)
    print("Predict value:", pred_y)
    print("True  value:", y_val)
