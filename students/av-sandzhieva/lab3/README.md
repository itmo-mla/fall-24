# Лабораторная работа 3
Ссылка на датасет [https://www.kaggle.com/datasets/saramah/diabets-data-set](https://www.kaggle.com/competitions/titanic/data) - задача предсказания кто из пассажиров выжил после кораблекрушения "Титаника".

## Визуализация данных
Визуализацию данных проводим с использованием методов уменьшения размерности: t-SNE (t-distributed Stochastic Neighbor Embedding) и PCA (Principal Component Analysis). Строим диаграммы рассеивания: 
<br>
![image](https://github.com/user-attachments/assets/68154e22-fbbd-4165-b8fb-e65647aab9cd)



## Решение
### **Критериии разбиения**: 
#### **Энтропия**
Энтропия измеряет неопределенность распределения. Например:
Если в группе пассажиров 50% выжили и 50% нет, то энтропия максимальна.
Если 100% выжили или 100% не выжили, то энтропия равна 0.

```python
def entropy(y):
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)
```

#### **Критерий Донского**
Оценивает, насколько уменьшится дисперсия переменной y при разбиении данных по значению определенного признака.

```python
def donskoy_criterion(X, y):
    n = len(y)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (y[i] != y[j]) and (X[i] != X[j]):
                count += 1
    return count
```

#### **Критерий энтропии для многоклассового разбиения**
Оценивает уменьшение энтропии после разбиения на группы.

```python
def multiclass_entropy_criterion(X_column, y):
    counts = Counter(y)
    total_count = len(y)
    weighted_entropy = 0

    for c in counts:
        Pc = counts[c] / total_count
        p = len(X_column[X_column == c]) / total_count
        if p > 0:
            weighted_entropy += Pc * (-p * np.log2(p))

    return weighted_entropy
```

---

#### **Как строится дерево?**

- Алгоритм `id3` рекурсивно выбирает лучший признак для разбиения на каждом этапе.
- Лучшая характеристика определяется на основе выбранного критерия (энтропия или Донской).
- Когда в узле остается только один класс (`len(np.unique(y)) == 1`), узел становится листом дерева.

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
#### **Прунинг**
Функция prune_tree предназначена для уменьшения сложности дерева решений, удаляя узлы, которые не вносят значительного вклада в предсказания. Это помогает избежать переобучения и улучшает обобщающую способность модели.

```python
def prune_tree(tree, min_samples=5):
    if not isinstance(tree, dict):
        return tree

    pruned_tree = {}
    for key in tree:
        subtree = tree[key]
        if isinstance(subtree, dict):
            subtree = {subkey: prune_tree(subval, min_samples) for subkey, subval in subtree.items()}

            # Если все значения поддерева ведут к одному результату
            leaf_values = list(subtree.values())
            if all(isinstance(leaf, dict) == False and leaf == leaf_values[0] for leaf in leaf_values):
                return leaf_values[0]
        pruned_tree[key] = subtree

    return pruned_tree
```
---

#### **Предсказание**
Функция `predict` рекурсивно проходит по дереву, используя значения признаков пассажира, чтобы определить результат.

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

### Метрики
```
Custom Classification Report entropy:
              precision    recall  f1-score   support

           0       0.76      0.94      0.84       105
           1       0.88      0.57      0.69        74

    accuracy                           0.79       179
   macro avg       0.82      0.76      0.76       179
weighted avg       0.81      0.79      0.78       179


Custom Classification Report donskoy:
              precision    recall  f1-score   support

           0       0.69      0.91      0.78       105
           1       0.77      0.41      0.53        74

    accuracy                           0.70       179
   macro avg       0.73      0.66      0.66       179
weighted avg       0.72      0.70      0.68       179


Библиотечная Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.78      0.81       105
           1       0.72      0.78      0.75        74

    accuracy                           0.78       179
   macro avg       0.78      0.78      0.78       179
weighted avg       0.79      0.78      0.78       179
```

### Матрицы ошибок
```
Custom Confusion Matrix entropy:
[[99  6]
 [32 42]]

Custom Confusion Matrix donskoy:
[[96  9]
 [44 30]]

Библиотечная Confusion Matrix:
[[82 23]
 [16 58]]
```

### Время 
Время работы (энтропия): 0.03813028335571289
Время работы (Донской): 0.8285832405090332
Время работы (библиотечное дерево): 0.0030066967010498047

## Выводы

1. **Точность (Precision)**:
   - Для класса 0, модель с энтропией показывает хороший precision (0.76), но модель с Донским имеет более низкий precision (0.69)

2. **Полнота (Recall)**:
   - Полнота для класса 1 в модели с энтропией (0.57) значительно ниже, чем у класса 0 (0.94). 
3. **F1-меры**:
   - F1-меры для класса 1 в модели с энтропией (0.69) и Донским (0.53) показывают, что модели имеют трудности с балансом между точностью и полнотой. 

4. **Матрицы ошибок**:
   - В матрицах ошибок видно, что кастомные модели имеют больше ложных отрицательных срабатываний для класса 1.

5. **Время**:
   - Более быстрым решением является решение "из коробки", но многоклассовый энтропийный критерий ненамного уступает, скорее всего из-за маленького датасета.

## Заключение
На основе проведенного анализа можно сделать вывод, что хотя кастомные модели показывают хорошие результаты для класса 0, они сталкиваются с проблемами при классификации класса 1. Библиотечная модель демонстрирует более сбалансированные результаты и может быть предпочтительнее для задач, требующих высокой точности и полноты для обоих классов.


### Примечания:

- **Стандартизация и балансировка классов**:
  - Применены стандартизация признаков и балансировка классов с помощью SMOTE для устранения дисбаланса между классами. Это позволило улучшить результаты по точности для класса 1, но кастомная модель все еще сталкивается с проблемами недоклассификации класса 1.
