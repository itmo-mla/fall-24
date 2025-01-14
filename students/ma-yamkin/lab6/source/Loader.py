import pandas as pd
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self):
        data = pd.read_csv('bikes_rent.csv')

        data = data.dropna()

        features = data.drop(columns=['cnt'], axis=1)
        target = data['cnt']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.3, random_state=12)
