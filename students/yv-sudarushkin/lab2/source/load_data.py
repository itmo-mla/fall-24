import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_fish():
    df = pd.read_csv('cv/fish_data.csv')
    return df


def encod(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    scaler_X = MinMaxScaler()
    numeric_cols = df.drop('species', axis=1).select_dtypes(include=[float, int]).columns
    df[numeric_cols] = scaler_X.fit_transform(df[numeric_cols])
    df['species'] = label_encoder.fit_transform(df['species'])
    return df
