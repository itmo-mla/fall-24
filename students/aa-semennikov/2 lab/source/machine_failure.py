import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def read_machine_failure():
    data = pd.read_csv("dataset/machine_failure_dataset.csv")
    #print(data['Failure_Risk'].value_counts())


    ###############################################################

    # # Подсчитаем количество экземпляров каждого класса
    # class_counts = data['Failure_Risk'].value_counts()
    # num_class_1 = class_counts[1]
    # num_class_0 = class_counts[0]

    # num_to_remove = num_class_0 - num_class_1
    # # Случайным образом выберем экземпляры класса 0 для удаления
    # class_0_indices = data[data['Failure_Risk'] == 0].index.to_list()
    # indices_to_remove = pd.Series(class_0_indices).sample(num_to_remove, random_state=42) 
    # data = data.drop(indices_to_remove)
    # print(data['Failure_Risk'].value_counts()) # теперь 50:50

    ###############################################################
    
    le = LabelEncoder()
    data['Machine_Type'] = le.fit_transform(data['Machine_Type'])
    scaler = StandardScaler()
    for c in ['Temperature', 'Vibration', 'Power_Usage', 'Humidity']:
        data[[c]] = scaler.fit_transform(data[[c]])
    return data[['Temperature', 'Vibration', 'Power_Usage', 'Humidity', 'Machine_Type']].to_numpy(), data[['Failure_Risk']].to_numpy()


if __name__ == "__main__":
    print(read_machine_failure()[0])