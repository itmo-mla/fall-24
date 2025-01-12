import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_data(filename):
    le = LabelEncoder()

    data = pd.read_csv(filename, index_col='Id')

    y = le.fit_transform(data['Species'])
    y = np.where(y == 0, -1, y)
    y = np.expand_dims(y, axis=1)

    del data['Species']
    X = data.to_numpy()
    return X, y