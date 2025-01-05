import pandas as pd
import ID3
import utils
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

def evaluate_tree(tree, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    end_time = time.time()

    r2 = r2_score(y_test, y_pred)

    print(f"Time taken {name}: {end_time - start_time:.4f} seconds")
    print(f"R^2 ({name}): {r2:.4f}")
    print("\n")

    utils.visualize_tree(tree.get_tree(), dir="regression", name=name)



df = pd.read_csv('./datasets/boston.csv')
print(df.head(5))



X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)





max_depth = 9

tree = ID3.Regressor(criteria='entropy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Entropy', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='entropy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Entropy Pruned', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='donskoy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Donskoy', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='donskoy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Donskoy Pruned', X_train, X_test, y_train, y_test)


tree = ID3.Regressor(criteria='mse', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'MSE', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='mse', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'MSE Pruned', X_train, X_test, y_train, y_test)

tree = DecisionTreeRegressor(criterion="squared_error", max_depth=max_depth, random_state=33)
start_time = time.time()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
end_time = time.time()

r2 = r2_score(y_test, y_pred)

print(f"Time taken SKLEARN: {end_time - start_time:.4f} seconds")
print(f"R^2 (Sklearn): {r2:.4f}")
print("\n")

feature_names = X.columns.tolist()
utils.visualize_tree_sklearn(tree, dir="regression", name="SklearnTree", feature_names=feature_names, is_regression=True)






df = utils.inject_nan_values(df, 0.05, 'MEDV')
print("Пропуски")
print(df.isnull().sum())

X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)





max_depth = 9

tree = ID3.Regressor(criteria='entropy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Entropy with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='entropy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Entropy Pruned with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='donskoy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Donskoy with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='donskoy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Donskoy Pruned with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='mse', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'MSE with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Regressor(criteria='mse', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'MSE Pruned with NAN', X_train, X_test, y_train, y_test)

tree = DecisionTreeRegressor(criterion="squared_error", max_depth=max_depth, random_state=33)
start_time = time.time()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
end_time = time.time()

r2 = r2_score(y_test, y_pred)

print(f"Time taken SKLEARN  with NAN: {end_time - start_time:.4f} seconds")
print(f"R^2 (Sklearn with NAN): {r2:.4f}")
print("\n")

feature_names = X.columns.tolist()
utils.visualize_tree_sklearn(tree, dir="regression", name="SklearnTree with NAN", feature_names=feature_names, is_regression=True)
