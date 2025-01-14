import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read(filename):
    le = LabelEncoder()
    df = pd.read_csv(filename, index_col='Order_ID')
    df = df.dropna()
    
    y = df['Delivery_Time_min'].to_numpy().astype(np.float32)
    del df['Delivery_Time_min']
    
    X = df
    X['Weather'] = le.fit_transform(X['Weather'])
    X['Traffic_Level'] = le.fit_transform(X['Traffic_Level'])
    X['Time_of_Day'] = le.fit_transform(X['Time_of_Day'])
    X['Vehicle_Type'] = le.fit_transform(X['Vehicle_Type'])

    X = X.to_numpy()
    return X, y