import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Loader:
    def __init__(self):
        self.data = pd.read_csv(r"C:\Users\makso\Desktop\АМО\linear\iris.csv")[0:100]

        le = preprocessing.LabelEncoder()
        self.data['species'] = le.fit_transform(self.data['species'])

        X = self.data.drop(['species'], axis=1)
        y = self.data['species']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=24)

