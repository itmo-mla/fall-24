import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_hero(filename):
    le = LabelEncoder()
    data = pd.read_csv(filename)
    df = data.drop(['name', 'url', 'full-name', 'relatives', 'group-affiliation', 'aliases', 'first-appearance', 'place-of-birth', 'occupation', 'base'], axis=1)
    alter_egos = []
    for d in df['alter-egos']:
        if d == 'No alter egos found.':
            alter_egos.append(0)
        elif pd.isna(d):
            alter_egos.append(d)
        else:
            alter_egos.append(1)
    df['alter-egos'] = alter_egos

    weight = []
    for d in df['weight']:
        val, metric = d.split(', ')[1].replace('\'', '').replace(',','').replace(']', '').split(' ')
        if int(val) == 0:
            weight.append(np.nan)
        else:
            if metric == 'kg':
                weight.append(float(val))
            elif metric == 'tons':
                weight.append(float(val) * 1000)
    df['weight'] = weight

    height = []
    for d in df['height']:
        delim = d.split(', ')[1].replace('\'', '').replace(',','').replace(']', '').split(' ')
        if len(delim) == 1:
            height.append(float('inf'))
        else:
            val, metric = delim
            if metric == 'cm':
                height.append(float(val))
            elif metric == 'meters':
                height.append(float(val) * 100)
    df['height'] = height
    df['height'] = np.where(df['height'] == 0.0, np.nan, df['height'])

    df['publisher'] = le.fit_transform(df['publisher'])
    df['publisher'] = np.where(df['publisher'] == len(le.classes_) - 1, np.nan, df['publisher'])

    df['alignment'] = le.fit_transform(df['alignment'])
    df['alignment'] = np.where(df['alignment'] == np.where(le.classes_ == '-')[0][0], np.nan, df['alignment'])


    df['gender'] = le.fit_transform(df['gender'])
    df['gender'] = np.where(df['gender'] == np.where(le.classes_ == '-')[0][0], np.nan, df['gender'])

    df['hair-color'] = np.where(df['hair-color'] == 'black', 'Black', df['hair-color'])
    df['hair-color'] = np.where((df['hair-color'] == 'Strawberry Blond') | (df['hair-color'] == 'blond'), 'Blond', df['hair-color'])
    hair = []
    for d in df['hair-color']:
        if len(d.split(' / ')) > 1:
            hair.append(d.split(' / ')[1])
        else:
            hair.append(d)
    df['hair-color'] = hair

    df['hair-color'] = le.fit_transform(df['hair-color'])
    df['hair-color'] = np.where(df['hair-color'] == np.where(le.classes_ == '-')[0][0], np.nan, df['hair-color'])

    df['eye-color'] = np.where(df['eye-color'] == 'brown', 'Brown', df['eye-color'])
    df['eye-color'] = np.where(df['eye-color'] == 'blue', 'Blue', df['eye-color'])
    color = []
    for d in df['eye-color']:
        if len(d.split(' / ')) > 1:
            color.append(d.split(' / ')[1])
        else:
            color.append(d)
    df['eye-color'] = color

    df['eye-color'] = le.fit_transform(df['eye-color'])
    df['eye-color'] = np.where(df['eye-color'] == np.where(le.classes_ == '-')[0][0], np.nan, df['eye-color'])


    aliens = ['Icthyo Sapien', 'Ungaran', 'Xenomorph XX121', 'Yoda\'s species', 'Frost Giant', 'Kryptonian', 'Dathomirian Zabrak', 'Symbiote', 'Czarnian', 'Asgardian', 'Bizarro', 'Kakarantharaian', 'Zen-Whoberian', 'Strontian', 'Gungan', 'Bolovaxian', 'Rodian', 'Flora Colossus', 'Martian', 'Spartoi', 'Luphomoid', 'Yautja', 'Talokite', 'Korugaran', 'Tamaranean']
    for a in aliens:
        df['race'] = np.where(df['race'] == a, 'Alien', df['race'])

    paranormal = ['Vampire', 'Demon', 'Zombie', 'Parademon', 'Neyaphem']
    for p in paranormal:
        df['race'] = np.where(df['race'] == p, 'Paranormal', df['race'])

    mech = ['Android', 'Cyborg']
    for m in mech:
        df['race'] = np.where(df['race'] == m, 'Mech', df['race'])

    mutant = ['Human / Radiation', 'Metahuman', 'Alpha', 'Atlantean', 'Amazon', 'Mutant / Clone', 'Human / Cosmic', 'Inhuman', 'Human / Altered', 'Saiyan']
    for m in mutant:
        df['race'] = np.where(df['race'] == m, 'Mutant', df['race'])

    god = ['Maiar', 'Demi-God', 'God / Eternal', 'Eternal', 'New God', 'Black Racer']
    for g in god:
        df['race'] = np.where(df['race'] == g, 'God', df['race'])

    human = ['Human / Clone', 'Clone']
    for h in human:
        df['race'] = np.where(df['race'] == h, 'Human', df['race'])

    half_human = ['Human-Vulcan', 'Human-Spartoi', 'Human-Vuldarian', 'Human-Kree']
    for hh in half_human:
        df['race'] = np.where(df['race'] == hh, 'HalfHumanAlien', df['race'])

    df['race'] = np.where(df['race'] == 'Planet', 'Cosmic Entity', df['race'])

    animals = ['Gorilla', 'Kaiju']
    for a in animals:
        df['race'] = np.where(df['race'] == a, 'Animal', df['race'])

    df['race'] = le.fit_transform(df['race'])
    df['race'] = np.where(df['race'] == len(le.classes_) - 1, np.nan, df['race'])

    y = df['alter-egos'].to_numpy()
    del df['alter-egos']

    X = df.to_numpy()

    return X, y

def read_cars(filename):
    le = LabelEncoder()
    df = pd.read_csv(filename)

    df['model'] = le.fit_transform(df['model'])
    df['brand'] = le.fit_transform(df['brand'])
    df['fuel_type'] = le.fit_transform(df['fuel_type'])
    df['engine'] = le.fit_transform(df['engine'])
    df['transmission'] = le.fit_transform(df['transmission'])
    df['ext_col'] = le.fit_transform(df['ext_col'])
    df['int_col'] = le.fit_transform(df['int_col'])

    clean = []
    for d in df['clean_title']:
        if d == 'Yes':
            clean.append(1)
        else:
            clean.append(d)
    df['clean_title'] = clean

    accident = []
    for d in df['accident']:
        if d == 'At least 1 accident or damage reported':
            accident.append(1)
        elif d == 'None reported':
            accident.append(0)
        else:
            accident.append(d)
    df['accident'] = accident

    milage = []
    for d in df['milage']:
        mil = d.split()
        milage.append(int(mil[0].replace(',', '')))
    df['milage'] = milage

    price = []
    for d in df['price']:
        price.append(int(d.replace(',', '').replace('$', '')))
    df['price'] = price

    y = df['price'].to_numpy()
    del df['price']

    X = df.to_numpy()
    return X, y


if __name__ == "__main__":
    X, y = read_hero('dataset/superheroes_data.csv')