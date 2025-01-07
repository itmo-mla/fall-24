import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

# Класс, который в последствии добавляется в словарь для удобного выбора
class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        childs=None,
        value=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.childs = childs
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    


class DecisionTree:
    def __init__(self, max_depth=10, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None


    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)

    def calck_unic(self, a: list):
        keys = set(a)
        return {key: len(a[a == key]) for key in keys}

    def predict(self, X):
        return np.array([self.travers_tree(x, self.tree) for x in X])

    def entropy(self, y: list):
        hist = self.calck_unic(y)
        hist = {key: val / len(y) for key, val in hist.items()}
        info = -np.sum([p * np.log2(p) for p in hist.values()])

        return info

    def information_gain(self, X_column: list, y: list):
        if len(set(y)) == 1:
            return 0

        X_column = np.array(X_column)
        X_column = X_column[~np.isnan(X_column)]
        n = len(y)
        # info(T)
        parent = self.entropy(y)
        uitems = self.calck_unic(X_column)
        info_x = np.sum(
            [val / n * self.entropy(y[X_column == key]) for key, val in uitems.items()]
        )
        split_info = -np.sum(
            [val / n * np.log2(val / n) for val in uitems.values() if val > 0]
        )

        if split_info != 0:
            return (parent - info_x) / split_info, list(uitems.keys())
        else:
            return 0, list(uitems.keys())

    def most_common(self, y):
        labels = self.calck_unic(y)
        return max(labels, key=labels.get)

    def best_split(self, X, y):
        best_feature = None
        best_gain = -1
        uitems = []
        for i in range(X.shape[1]):
            gain, now_uitems = self.information_gain(X[:, i], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                uitems = now_uitems + []

        return best_feature, uitems

    def grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(self.calck_unic(y))

        if n_samples <= self.min_samples or depth >= self.max_depth or n_labels <= 1:
            return Node(value=self.most_common(y))

        best_feature, ukeys = self.best_split(X, y)

        # В словаре содержатся не словари, а Node По сути, словарь содержит ссылки на объекты, а нужен он для более удобной навигации.
        childs = {
            key: self.grow_tree(
                X[X[:, best_feature] == key],
                y[X[:, best_feature] == key],
                depth=depth + 1,
            )
            for key in ukeys
        }

        return Node(best_feature, childs=childs)

    def travers_tree(self, x, tree):
        if tree is None:
            return None
        
        elif tree.is_leaf_node():
            return tree.value

        return self.travers_tree(
            x,
            tree.childs.get(
                x[tree.feature], tree.childs.get(list(tree.childs.keys())[0])  #None
            ),
        )
    

if __name__ == "__main__":

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)

    transform = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    dataset["class"] = dataset["class"].apply(lambda x: transform[x])




    x = dataset[dataset.columns[:-1]].to_numpy()[:, :2]
    y = dataset["class"].to_numpy()


    from time import time

    start = time()
    clf = DecisionTree(max_depth=2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    clf.fit(x_train, y_train)

    otv = clf.predict(x_test)
    score = f1_score(y_test, otv, average="micro")
    print(f"f1_score {score}")
    print(f"calc time is: {time()-start}")


    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)

    start = time()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = f1_score(y_test, y_pred, average="micro")
    print(f"f1_score sklearn {score}")
    print(f"sklearn calc time is: {time()-start}")