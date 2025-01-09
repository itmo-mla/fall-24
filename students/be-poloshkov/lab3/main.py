import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier, \
	DecisionTreeRegressor as SKDecisionTreeRegressor
from tree import DecisionTree, DecisionTreeRegressor


def test_regressor(instance):
	instance.fit(reg_X_train, reg_y_train)
	y_pred = instance.predict(reg_X_test)
	print('Mean Squared Error')
	print(mse(reg_y_test, y_pred))
	plot_regression(y_pred, reg_y_test)

def test_classifier(instance, post_prune: bool):
	instance.fit(clf_X_train, clf_y_train)
	if post_prune:
		instance.post_prune(clf_X_test, clf_y_test)
		instance.print_tree()
	y_pred = instance.predict(clf_X_test)
	print(set(clf_y_test) - set(y_pred))
	print('Classification Report')
	print(classification_report(clf_y_test, y_pred))


def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()


def plot_regression(y_pred, y_true, lim=100):
	x = list(range(len(y_pred)))[:lim]
	plt.plot(x, y_pred[:lim], label='prediction')
	plt.plot(x, y_true[:lim], label='true')
	plt.legend()
	plt.show()


def prepare_dataset(df: pd.DataFrame, target: str) -> tuple:
	X, y = df.drop(columns=[target]), df[target]
	categorical_features = X.columns[(X.dtypes == object)]

	for c in categorical_features:
		X[c] = LabelEncoder().fit_transform(X[c])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	return X_train, X_test, y_train, y_test


if __name__ == '__main__':
	df_titanic = pd.read_csv('titanic.csv')
	df_titanic.head()

	df_salary = pd.read_csv('salary.csv')
	df_salary.head()

	clf_X_train, clf_X_test, clf_y_train, clf_y_test = prepare_dataset(df_titanic,'Survived')
	reg_X_train, reg_X_test, reg_y_train, reg_y_test = prepare_dataset(df_salary, 'Salary')

	# Classification

	inf = 1e10
	max_depth = 5
	max_leafs = 15

	classifiers = (
		('Self-written, no reduction: gini',
		 DecisionTree(max_depth=inf, max_leafs=inf, min_samples_split=0),
		 False),

		('Self-written, no reduction: donskoy',
		 DecisionTree(max_depth=inf, max_leafs=inf, min_samples_split=0, criterion='donskoy'),
		 False),

		('Self-written, pre-reduction: gini',
		 DecisionTree(max_depth=max_depth, max_leafs=max_leafs, min_samples_split=0),
		 False),

		('Self-written, pre-reduction: donskoy',
		 DecisionTree(max_depth=max_depth, max_leafs=max_leafs, min_samples_split=0, criterion='donskoy'),
		 False),

		('Self-written, post-reduction: gini',
		 DecisionTree(max_depth=inf, max_leafs=inf, min_samples_split=0),
		 True),

		('Self-written, post-reduction: donskoy',
		 DecisionTree(max_depth=inf, max_leafs=inf, min_samples_split=0, criterion='donskoy'),
		 True),

		('SK, no reduction: gini', SKDecisionTreeClassifier(), False),

		('SK, reduction: gini',
		 SKDecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leafs), False)
	)

	for label, clf, post_prune in classifiers:
		print(f'\t{label.upper()}')
		start = time.monotonic()
		test_classifier(clf, post_prune)
		print('Time', time.monotonic() - start)

	# Regression

	max_leafs = 10
	max_depth = 3

	regressors = (
		('Self-written, no reduction', DecisionTreeRegressor(max_depth=inf, max_leafs=inf, min_samples_split=0)),
		('Self-written, reduction', DecisionTreeRegressor(max_depth=max_depth, max_leafs=max_leafs, min_samples_split=0)),
		('SK, no reduction', SKDecisionTreeRegressor()),
		('SK, reduction', SKDecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leafs)),
	)

	for label, reg in regressors:
		print(f'\t{label}')
		start = time.monotonic()
		test_regressor(reg)
		print('Time', time.monotonic() - start)
