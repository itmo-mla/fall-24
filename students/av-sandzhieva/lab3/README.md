# Лабораторная работа 3
Ссылка на датасет [https://www.kaggle.com/datasets/saramah/diabets-data-set](https://www.kaggle.com/competitions/titanic/data) - задача предсказания кто из пассажиров выжил после кораблекрушения "Титаника".

## Визуализация данных
Визуализацию данных проводим с использованием методов уменьшения размерности: t-SNE (t-distributed Stochastic Neighbor Embedding) и PCA (Principal Component Analysis). Строим диаграммы рассеивания: 
<br>
![image](https://github.com/user-attachments/assets/45bb074b-505c-4907-8596-509397d8a801)


## Решение
### **Критериии разбиения**: 
   - Реализованы два критерия:
     - **Энтропия (`entropy`)**: Мера неопределенности. Чем меньше энтропия, тем более "чистое" разбиение.
     - **Критерий Донского (`donskoy_criterion`)**: Оценивает уменьшение дисперсии внутри группы после разбиения.

#### **Энтропия**
Энтропия измеряет неопределенность распределения. Например:
- Если в группе пассажиров 50% выжили и 50% нет, то энтропия максимальна.
- Если 100% выжили или 100% не выжили, то энтропия равна 0.

```python
def entropy(y):
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)
```

#### **Критерий Донского**
Оценивает, насколько уменьшится дисперсия переменной `y` при разбиении данных по значению определенного признака.

```python
def donskoy_criterion(X_column, y):
    values = np.unique(X_column)
    return np.var(y) - sum(len(y[X_column == value]) / len(y) * np.var(y[X_column == value]) for value in values)
```

#### **Критерий энтропии для многоклассового разбиения**
Оценивает уменьшение энтропии после разбиения на группы.

```python
def multiclass_entropy_criterion(X_column, y):
    values = np.unique(X_column)
    weighted_entropy = sum(len(y[X_column == value]) / len(y) * entropy(y[X_column == value]) for value in values)
    return entropy(y) - weighted_entropy
```

---

#### **Как строится дерево?**

- Алгоритм `id3` рекурсивно выбирает лучший признак для разбиения на каждом этапе.
- Лучшая характеристика определяется на основе выбранного критерия (энтропия или Донской).
- Когда в узле остается только один класс (`len(np.unique(y)) == 1`), узел становится листом дерева.

Пример узла:
```python
def id3(X, y, feature_names, criterion="entropy", max_depth=None, current_depth=0):
    # Базовые случаи
    if len(np.unique(y)) == 1 or len(feature_names) == 0 or (max_depth is not None and current_depth >= max_depth):
        return Counter(y).most_common(1)[0][0]

    gains = CRITERIA[criterion](X, y)
    best_feature_idx = np.argmax(gains)
    best_feature = feature_names[best_feature_idx]

    tree = {best_feature: {}}

    for value in np.unique(X[:, best_feature_idx]):
        subset_X = X[X[:, best_feature_idx] == value]
        subset_y = y[X[:, best_feature_idx] == value]
        
        subtree = id3(
            np.delete(subset_X, best_feature_idx, axis=1),
            subset_y,
            feature_names[:best_feature_idx] + feature_names[best_feature_idx + 1:],
            criterion=criterion,
            max_depth=max_depth,
            current_depth=current_depth + 1
        )
        tree[best_feature][value] = subtree

    return tree
```

---

#### **Предсказание**
Функция `predict` рекурсивно проходит по дереву, используя значения признаков пассажира, чтобы определить результат.

Пример:
```python
def predict(tree, X_sample, feature_names):
    if not isinstance(tree, dict):
        return tree  # Если узел — это результат, возвращаем его.
    root_node = next(iter(tree))
    feature_idx = feature_names.index(root_node)
    subtree = tree[root_node].get(X_sample[feature_idx], 0)  # Если значение не найдено, используем 0.
    return predict(subtree, X_sample, feature_names)
```

---
## Результаты

### Сustom Classification Report (Entropy и Donskoy):

| Метрика      | Класс 0 | Класс 1 | Среднее значение |
|--------------|---------|---------|------------------|
| **Precision**| 0.69    | 0.77    | 0.73             |
| **Recall**   | 0.91    | 0.41    | 0.66             |
| **F1-score** | 0.78    | 0.53    | 0.66             |
| **Accuracy** | 0.70    |         | 0.70             |

### Библиотечный Classification Report:

| Метрика      | Класс 0 | Класс 1 | Среднее значение |
|--------------|---------|---------|------------------|
| **Precision**| 0.80    | 0.73    | 0.76             |
| **Recall**   | 0.81    | 0.72    | 0.76             |
| **F1-score** | 0.81    | 0.72    | 0.76             |
| **Accuracy** | 0.77    |         | 0.77             |

### Custom Confusion Matrix(the same for both criterion):

|               | Класс 0 | Класс 1 |
|---------------|---------|---------|
| **Класс 0**   | 96      | 9       |
| **Класс 1**   | 44      | 30      |

### Библиотечная Confusion Matrix:

|               | Класс 0 | Класс 1 |
|---------------|---------|---------|
| **Класс 0**   | 85      | 20      |
| **Класс 1**   | 21      | 53      |

### Заключение:

- **Кастомное решение (Entropy и Donskoy)**:
  - Модель дает хорошие результаты по классу 0, но имеет проблемы с недоклассификацией класса 1 (Recall = 0.41). Это снижает F1-скор для класса 1 до 0.53, что указывает на необходимость улучшения классификации этого класса.
  
- **Библиотечное решение**:
  - Модель библиотеки показывает улучшенные результаты, особенно для класса 1 (точность и полнота выше, чем у кастомной модели), что приводит к более сбалансированным результатам и улучшению F1-меры.

### Примечания:

- **Стандартизация и балансировка классов**:
  - Применены **стандартизация** признаков и **балансировка классов** с помощью **SMOTE** для устранения дисбаланса между классами. Это позволило улучшить результаты по точности для класса 1 в библиотеки, но кастомная модель все еще сталкивается с проблемами недоклассификации класса 1.
