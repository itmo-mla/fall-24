import pandas as pd
import ID3
import utils
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female':0, 'Male':1})
print(df.head(5))


X = df.drop(['Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)


def evaluate_tree(tree, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    end_time = time.time()
    print(f"Time taken {name}: {end_time - start_time:.4f} seconds")
    print(classification_report(y_test, y_pred))
    print("Accuracy", accuracy_score(y_test, y_pred))
    print("\n")
    utils.visualize_tree(tree.get_tree(), dir="classification", name=name)

max_depth = 4

tree = ID3.Classifier(criteria='entropy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Entropy', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='entropy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Entropy Pruned', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='donskoy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Donskoy', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='donskoy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Donskoy Pruned', X_train, X_test, y_train, y_test)


tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=33)
start_time = time.time()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
end_time = time.time()
print(f"Time taken SKLEARN: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("\n")
feature_names = X.columns.tolist()
utils.visualize_tree_sklearn(tree, dir="classification", name="SklearnTree", feature_names=feature_names)



df = utils.inject_nan_values(df, 0.05, 'Survived')
print("Пропуски")
print(df.isnull().sum())

X = df.drop(['Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

max_depth = 4

tree = ID3.Classifier(criteria='entropy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Entropy with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='entropy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Entropy Pruned with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='donskoy', max_depth=max_depth, prune=False)
evaluate_tree(tree, 'Donskoy with NAN', X_train, X_test, y_train, y_test)

tree = ID3.Classifier(criteria='donskoy', max_depth=max_depth, prune=True)
evaluate_tree(tree, 'Donskoy Pruned with NAN', X_train, X_test, y_train, y_test)


tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=33)
start_time = time.time()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
end_time = time.time()
print(f"Time taken SKLEARN with NAN: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("\n")
feature_names = X.columns.tolist()
utils.visualize_tree_sklearn(tree, dir="classification", name="SklearnTree with NAN", feature_names=feature_names)
