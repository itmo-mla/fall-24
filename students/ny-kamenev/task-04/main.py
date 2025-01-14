import classifier
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier


df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female': 0, 'Male': 1})
df['Survived'] = df['Survived'].map({0: -1, 1: 1})
print(df.head(5))

sns.set(rc={'figure.figsize': (15, 8)})
plt.figure(figsize=(15, 8))
heatmap = sns.heatmap(df.corr(), annot=True, linewidths=3, cbar=False)
plt.savefig('./img/heatmap.png')
plt.close()

X = df.drop(['Survived'], axis=1)
y = df['Survived']

scaler = preprocessing.MinMaxScaler()
X = np.array(scaler.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
num_features = X.shape[1]

# ========================================================================================
experiment = 'GD'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=None,
                 momentum=None,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)


c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))

# ========================================================================================
experiment = 'NAG'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=None,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)


c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'L2'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=0.01,
                 momentum=None,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)


c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'Fast'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=0.01,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=True,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=100)

c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'Margin_based'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=0.01,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=True)
c.fit(X_train, y_train,
      num_iterations=1000)



c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'Correlation'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='correlation',
                 l2=0.01,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)



c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'Multistart'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='multistart',
                 l2=0.01,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)


c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


# ========================================================================================
experiment = 'NAG+L2'
print("\n\n\nExperiment ", experiment)

start_time = time.time()
c = classifier.Classifier(num_features,
                 learning_rate=0.01,
                 start_weights='random',
                 l2=0.01,
                 momentum=0.05,
                 forget_temp=0.001,
                 fast_gd=False,
                 margin_based_sampling=False)
c.fit(X_train, y_train,
      num_iterations=1000)

c.plot_results(X_test, y_test, experiment)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))

# ========================================================================================
experiment = 'SKLEARN'
print("\n\n\nExperiment ", experiment)
start_time = time.time()
c = SGDClassifier()
c.fit(X_train, y_train)
pred = c.predict(X_test)
end_time = time.time()
print(f"Time taken {experiment}: {end_time - start_time:.4f} seconds")
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
