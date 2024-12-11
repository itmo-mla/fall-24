from LinearClassificator import LinearClassifier
from Loader import Loader
from sklearn.metrics import accuracy_score


loader = Loader()
X_train, X_test, y_train, y_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test


classifier = LinearClassifier()
g = classifier.margin(X_train, y_train)
classifier.visualize(X_train, y_train, g)

classifier(lr=0.01, a=0.01, lambda_reg=0.0001, delta=0.0000001, optimizer='sgd', t=0.9, M=False, w='multi')
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print(accuracy_score(y_test, pred))
