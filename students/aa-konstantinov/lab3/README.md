# Лабораторная 3 **Логическая классификация**

1. **Логическая классификация**  
    Реализован в файле `ID3.py`.



## Используемые Датасеты

Для тестирования и оценки работы алгоритмов был использован следующий датасет:

 - **Kaggle - Breast Cancer Wisconsin**  
   [Ссылка на датасет](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

 - **Iris - load_iris**

 - **Stars - https://github.com/YBIFoundation/Dataset/raw/main/Stars.csv**

 - **Boston - https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset**


## Метрики

### Результаты классификации

| Dataset     | Метод                                                       | Препроцессинг                | Accuracy до прунинга | Accuracy после прунинга | CPU times (total) | Wall time |
|-------------|-------------------------------------------------------------|------------------------------|----------------------|-------------------------|-------------------|-----------|
| load_iris   | ID3 (gain_type='entropy', min_samples_leaf=4, max_depth=10) | Без прунинга / С прунингом   | 0.9473684210526315   | 0.9736842105263158      | 15.7 ms           | 15.1 ms   |
| Stars       | ID3 (gain_type='entropy', min_samples_leaf=4, max_depth=10) | Без прунинга / С прунингом   | 1.0                  | 1.0                     | 32.3 ms           | 32.2 ms   |
| Stars       | DecisionTreeClassifier(criterion='entropy', random_state=42) | -                            | 1.0                  | -                       | 1.76 ms           | 1.32 ms   |

### Результаты регрессии

| Dataset | Метод                                                                        | Препроцессинг    | MSE    | MAE     | Wall time |
|---------|-------------------------------------------------------------------------------|------------------|--------|-------|-----------|
| boston  | DecisionTreeRegressor(random_state=42, max_depth=300, criterion='squared_error') | Без прунинга     | 543.08| 4.11 | 1.88 ms   |
| boston  | ID3 (ID3_type='regression', max_depth=300)                                   | После прунинга/Без прунинга   | 550.44 | 5.13  | 15.9 ms  |


