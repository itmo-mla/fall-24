import pandas as pd
from sklearn import preprocessing

def read_fish():
    data = pd.read_csv('dataset/fish_data.csv')
    le = preprocessing.LabelEncoder()
    le.fit(data.species)
    data['species'] = le.transform(data.species)
    return data[['length', 'weight', 'w_l_ratio']].to_numpy(), data['species'].to_numpy()

if __name__ == '__main__':
    data, label = read_fish()