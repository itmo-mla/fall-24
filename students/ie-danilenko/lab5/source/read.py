import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_student(filepath):
    encoder = LabelEncoder()
    scaler = StandardScaler()

    data = pd.read_csv(filepath, index_col='id')
    data['Gender'] = encoder.fit_transform(data['Gender'])
    data['City'] = encoder.fit_transform(data['City'])
    data['Profession'] = encoder.fit_transform(data['Profession'])
    data['Sleep Duration'] = encoder.fit_transform(data['Sleep Duration'])
    data['Dietary Habits'] = encoder.fit_transform(data['Dietary Habits'])
    data['Degree'] = encoder.fit_transform(data['Degree'])
    data['Have you ever had suicidal thoughts ?'] = encoder.fit_transform(data['Have you ever had suicidal thoughts ?'])
    data['Family History of Mental Illness'] = encoder.fit_transform(data['Family History of Mental Illness'])
    y = encoder.fit_transform(data['Depression'])
    del data['Depression']
    X = scaler.fit_transform(data.to_numpy())
    return X, np.where(y == 0, -1, 1)

if __name__ == "__main__":
    X, y = read_student('dataset/student.csv')