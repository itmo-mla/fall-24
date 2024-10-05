import pandas as pd
from sklearn import preprocessing

def read_fish(path):
    data = pd.read_csv(path)
    le = preprocessing.LabelEncoder()
    le.fit(data.species)
    data['species'] = le.transform(data.species)
    return data[['length', 'weight', 'w_l_ratio']].to_numpy(), data['species'].to_numpy()

if __name__ == '__main__':
    data, label = read_fish('dataset/fish.csv')
    print(data)
    print(label)