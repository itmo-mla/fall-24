import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

# task 1
label_encoder = preprocessing.LabelEncoder()
data = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')
data['State'] = label_encoder.fit_transform(data['State'])
print(data)

# task 1.2
plt.scatter(data['K&A'].to_numpy(), data['WT'].to_numpy())
plt.show()