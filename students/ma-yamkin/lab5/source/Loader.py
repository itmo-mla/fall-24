import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Loader:
    def __init__(self, data):
        if data == 'mashroom':
            self.data = pd.read_csv('2. Mushroom_dataset', delimiter = ",", names=['class-label','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])
            self.data = self.data.drop(['veil-type'], axis=1)

            le = preprocessing.LabelEncoder()
            for col in self.data.columns:
                self.data[col] = le.fit_transform(self.data[col])

            X = self.data.drop(['class-label'], axis=1)
            y = self.data['class-label']
        else:
            self.data = pd.read_csv(r"C:\Users\makso\Desktop\АМО\linear\iris.csv")[0:100]
            le = preprocessing.LabelEncoder()
            self.data['species'] = le.fit_transform(self.data['species'])

            X = self.data.drop(['species'], axis=1)
            y = self.data['species']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=24)

