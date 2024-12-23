import numpy as np
from read import read_cars
from sklearn.model_selection import train_test_split
from copy import deepcopy

class DecisionTreeRegression:
    def __init__(self, min_split):
        self.min_split =  min_split
        self.tree = None
        self.probabilities = None

        self.__max_probabilitiesbilities = -1

    def fit(self, X, y):
        not_nan_index = ~np.isnan(X).any(axis=1)
        X = X[not_nan_index]
        y = y[not_nan_index]

        _, counts = np.unique(y, return_counts=True)
        self.majority_reg = np.argmax(counts)
        self.probabilities = counts / len(y)
        
        self.tree = self._id3_regression(X, y)

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
            return (np.mean(y_val), None, None, None, left_probabilities)
        
    def __check_accuracy(self, tree, X_val, y_val):
        predictions = self._predict(X_val, tree)
        return np.mean(np.square(y_val - predictions))
    
    def pruning(self, X, y):
        self.tree = self.__prune_all_tree(self.tree, X, y)

    def _rmse(self, y):
        return np.sqrt(np.mean((y - np.mean(y)) ** 2))
    
    def calculate_rmse_reduction(self, y, mask):
        if len(y) == 0:
            return 0
        total_rmse = self._rmse(y)
        left_rmse = self._rmse(y[mask])
        right_rmse = self._rmse(y[~mask])
        
        weighted_rmse = (len(y[mask]) / len(y)) * left_rmse + (len(y[~mask]) / len(y)) * right_rmse
        return total_rmse - weighted_rmse

    def _calculate_information_gain(self, y, mask):
        total_entropy = self._entropy(y)
        left_entropy = self._entropy(y[mask])
        right_entropy = self._entropy(y[~mask])
        
        weighted_entropy = (len(y[mask]) / len(y)) * left_entropy + (len(y[~mask]) / len(y)) * right_entropy
        return total_entropy - weighted_entropy

    def _id3_regression(self, X, y):
        if X.shape[1] < self.min_split:
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return np.mean(y), None, None, probabilities, probabilities[0]
        
        if len(np.unique(y)) == 1:
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return np.mean(y), None, None, probabilities, probabilities[0]
        
        notnon_index = ~np.isnan(X).any(axis=1)
        X = X[notnon_index]
        y = y[notnon_index]
        
        best_feature = None
        best_threshold = None
        best_rmse_reduction = -np.inf
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                mask = X[:, feature_index] <= threshold
                
                rmse_reduction = self.calculate_rmse_reduction(y, mask)
                if rmse_reduction > best_rmse_reduction:
                    best_rmse_reduction = rmse_reduction
                    best_feature = feature_index
                    best_threshold = threshold
        
        if best_feature is None:
            _, counts = np.unique(y, return_counts=True)
            probabilitiesbilities = counts / len(y)
            return np.mean(y), None, None, probabilitiesbilities, probabilitiesbilities
        
        mask = X[:, best_feature] <= best_threshold
        left_subtree = self._id3_regression(X[mask], y[mask])
        right_subtree = self._id3_regression(X[~mask], y[~mask])
        left_probabilities = len(y[mask]) / len(y)
        
        return (best_feature, best_threshold, left_subtree, right_subtree, left_probabilities)

    def _predict_non(self, sample, tree, current_probabilitiesbilities):
        feature_index, threshold, left_subtree, _, left_probabilities = tree

        if threshold is None:
            if self.__max_probabilitiesbilities < current_probabilitiesbilities * feature_index:
                self.__max_probabilitiesbilities = current_probabilitiesbilities * feature_index
                return feature_index
            return self.majority_reg
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
    X, y = read_cars('dataset/used_cars.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    decision_tree = DecisionTreeRegression(2)
    decision_tree.fit(X_train, y_train)
    pred_y = decision_tree.predict(X_val).astype(y_val.dtype)
    print("Predict value:", pred_y)
    print("True  value:", y_val)

    pruned_tree = deepcopy(decision_tree)
    pruned_tree.pruning(X_val, y_val)
    pred_y = pruned_tree.predict(X_val).astype(y_val.dtype)
    print("Predict value:", pred_y)
    print("True  value:", y_val)
    