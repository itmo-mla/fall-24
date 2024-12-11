import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

# 1.1 - метки классов отсутствуют
# 1.3 - ленточные кластеры
# 1.4 - предполагаемое количество кластеров = 3

def read_weapons(plot = False):
    data = pd.read_csv("datasets/Skyrim_Weapons.csv").drop(['Name', 'Upgrade', 'Perk', 'Type', 'Speed'], axis=1).dropna()
    label_encoder = preprocessing.LabelEncoder()
    data['Category'] = label_encoder.fit_transform(data['Category'])    
    for c in data.columns:
        data[[c]] = preprocessing.StandardScaler().fit_transform(data[[c]])
    if plot == False:
        return data
    else:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.show()       

if __name__ == "__main__":
    read_weapons(plot = True)