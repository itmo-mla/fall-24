import numpy as np
from id3_regression import DecisionTreeRegression
from sklearn.model_selection import train_test_split
from read import read_hero
from sklearn.tree import DecisionTreeRegressor
from time import time
from copy import deepcopy

def mse(y_pred, y):
    return np.mean(np.square(y - y_pred))

if __name__ == '__main__':
    X, y = read_hero('dataset/superheroes_data.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_prune = X_val[0:X_val.shape[0]//2]
    X_val = X_val[X_val.shape[0]//2:]

    y_prune = y_val[0:y_val.shape[0]//2]
    y_val = y_val[y_val.shape[0]//2:]

    decision_tree = DecisionTreeRegression(2)
    decision_tree.fit(X_train, y_train)
    pruned_decision_tree = deepcopy(decision_tree)
    pruned_decision_tree.pruning(X_prune, y_prune)

    start_time = time()
    y_pred = decision_tree.predict(X_val).astype(y_val.dtype)
    print("Id3 Regression")
    print("\tTime", time() - start_time)
    print("\tMSE", mse(y_pred, y_val))
    
    start_time = time()
    y_pred = pruned_decision_tree.predict(X_val).astype(y_val.dtype)
    print("Id3 Prune Regression")
    print("\tTime", time() - start_time)
    print("\tMSE", mse(y_pred, y_val))

    etalon_tree = DecisionTreeRegressor()
    etalon_tree.fit(X_train, y_train, check_input=False)

    y_pred = etalon_tree.predict(X_val)

    print("Etalon tree")
    print("\tTime", time() - start_time)
    print("\tMSE", mse(y_pred, y_val))