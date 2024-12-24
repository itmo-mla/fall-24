# Лабораторная работа №3. Логическая классификация

## Описание

Целью данной лабораторной работы является реализация алгоритма построения бинарного дерева решений (ID3) для задач классификации и регрессии, сравнение его с эталонной реализацией из библиотеки `scikit-learn`, а также применение алгоритма обрезки дерева (pruning).

## Задание

1. Выбрать датасет для классификации и регрессии, удовлетворяющий следующим условиям:
   - Датасет содержит пропущенные значения (NaN).
   - В датасете есть как категориальные, так и количественные признаки.
2. Реализовать алгоритм построения бинарного дерева решений (ID3):
   - С использованием критерия Донского.
   - С использованием многоклассового энтропийного критерия.
3. Реализовать аналогичный алгоритм для задачи регрессии, используя среднеквадратичную ошибку (MSE) в качестве критерия неопределенности.
4. Реализовать алгоритм обрезки дерева для классификации (pruning).
5. Сравнить качество работы алгоритмов до и после обрезки дерева.
6. Сравнить время работы кастомных алгоритмов и эталонной реализации из `scikit-learn`.
7. Подготовить отчет с выводами.


## Предварительная обработка данных

Для выполнения задач классификации и регрессии используется предварительная обработка данных, включающая кодирование категориальных признаков и обработку пропущенных значений.

### Кодирование категориальных признаков

Для преобразования категориальных признаков в числовые используется `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

# Пример кодирования категориальных признаков
categorical = ['blue', 'dual_sim', 'four_g', 'm_dep', 'price_range', 'n_cores', 'three_g', 'touch_screen', 'wifi']
le = LabelEncoder()
for feature in categorical:
    df[feature] = le.fit_transform(df[feature])
```

Введение пропусков (NaN)
Для симуляции пропущенных значений в данных, случайные элементы заменяются на NaN:
```python
# Задаем количество пропусков
num_nan = int(X_train.shape[0] * 0.2)

# Получаем размеры данных
rows, cols = X_train.shape

# Заменяем случайные элементы на NaN
for _ in range(num_nan):
    random_row = np.random.randint(0, rows)
    random_col = np.random.randint(0, cols)
    X_train.iloc[random_row, random_col] = np.nan
```

## Реализация дерева решений

### Структура узла дерева

Для задачи классификации структура узла включает функции для разделения данных, вычисления порога и вероятностей классов. Пример реализации узла:

```python
class Node:
    def __init__(self, criteria, classes=None):
        self.criteria = criteria  # Критерий для вычисления прироста информации
        self.feature_idx = None   # Индекс признака для разделения
        self.predicat = None      # Порог разделения
        self.info_gain = 0        # Прирост информации
        self.prob_left = 1        # Вероятность принадлежности к левой ветви
        self.left = None          # Левый дочерний узел
        self.right = None         # Правый дочерний узел
        self.classes = classes    # Список всех классов
        self.prob = None          # Вероятности классов в узле
```

Для задачи регрессии узел вместо классов содержит предсказанное значение:
```python
class Node:
    def __init__(self, criteria):
        self.criteria = criteria  # Критерий для вычисления MSE
        self.feature_idx = None   # Индекс признака для разделения
        self.predicat = None      # Порог разделения
        self.info_gain = 0        # Прирост информации
        self.value = None         # Значение целевой переменной
        self.left = None          # Левый дочерний узел
        self.right = None         # Правый дочерний узел
```

## Функция построения дерева
Для классификации:

```python
def decision_tree_classifier(X, y, criteria, classes, current_depth=0, max_depth=4):
    node = Node(criteria=criteria, classes=classes)
    (X_left, y_left), (X_right, y_right) = node.fit(X, y)  # Разделение данных
    if current_depth < max_depth and len(np.unique(y)) > 1:  # Рекурсивное построение
        node.left = decision_tree_classifier(X_left, y_left, criteria, classes, current_depth + 1, max_depth)
        node.right = decision_tree_classifier(X_right, y_right, criteria, classes, current_depth + 1, max_depth)
    return node
```
Для регрессии:
```python
def decision_tree_regressor(X, y, criteria, current_depth=0, max_depth=4):
    node = Node(criteria=criteria)
    (X_left, y_left), (X_right, y_right) = node.fit(X, y)  # Разделение данных
    if current_depth < max_depth and len(y) > 1:  # Рекурсивное построение
        if len(y_left) > 0:
            node.left = decision_tree_regressor(X_left, y_left, criteria, current_depth + 1, max_depth)
        if len(y_right) > 0:
            node.right = decision_tree_regressor(X_right, y_right, criteria, current_depth + 1, max_depth)
    return node
```

## Критерии для разделения
**Критерий Донского (классификация)**
```python
def donskoy_criteria(X, y):
    max_info_gain = -1
    best_weight = None
    for predicat in sorted(np.unique(X))[:-1]:
        p = X > predicat
        info_gain = np.sum((p[:, None] != p) & (y[:, None] != y))
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 4)
            best_weight = predicat
    return max_info_gain, best_weight
```

**Многоклассовый энтропийный критерий (классификация)**
```python
def multiclass_entropy_criterion(X, y):
    max_info_gain = -1
    best_weight = None
    l = len(X)
    p_cls_total = np.array([np.sum(y == cls) for cls in np.unique(y)])
    for predicat_weight in sorted(np.unique(X))[:-1]:
        p = np.sum(X > predicat_weight)
        right_entropy = ...
        left_entropy = ...
        info_gain = ...
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 4)
            best_weight = predicat_weight
    return max_info_gain, best_weight
```
**MSE-критерий (регрессия)**
```python
def mse_criteria(X, y):
    max_info_gain = -1
    best_weight = None
    for predicat_weight in sorted(np.unique(X))[:-1]:
        mask_right = X > predicat_weight
        mask_left = ~mask_right
        y_right = y[mask_right]
        y_left = y[mask_left]
        total_uncertainty = ...
        right_uncertainty = ...
        left_uncertainty = ...
        info_gain = ...
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 4)
            best_weight = predicat_weight
    return max_info_gain, best_weight

```

## Обрезка дерева (Pruning)

Обрезка дерева позволяет уменьшить его сложность, улучшив обобщающую способность модели. Для этого анализируются ошибки текущего узла и возможных упрощений (оставление одного поддерева или превращение узла в лист).

### Пример реализации обрезки

Алгоритм обрезки дерева классификации:

```python
def prune_classification_tree(node, X, y):
    """
    Обрезка дерева решений для классификации.
    
    Args:
        node: Узел дерева решений (объект класса Node).
        X (numpy.ndarray): Признаки.
        y (numpy.ndarray): Целевая переменная.
    
    Returns:
        Узел после возможной обрезки.
    """
    def compute_errors(node, X, y):
        """
        Вычисление ошибок для текущего узла.
        """
        if node.left is None and node.right is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            most_freq_class = unique_classes[np.argmax(counts)]
            predicted_class = np.argmax(node.prob)
            return (
                np.mean(y != predicted_class),  # Ошибка текущего узла
                float('inf'),  # Ошибка, если оставить только левое поддерево
                float('inf'),  # Ошибка, если оставить только правое поддерево
                np.mean(y != most_freq_class)  # Ошибка, если узел превратить в лист
            )
        
        predictions = np.array([np.argmax(node.predict_single(row)) for row in X])
        err_curr = np.mean(y != predictions)
        err_left = np.inf
        err_right = np.inf
        
        if node.left is not None:
            left_predictions = np.array([np.argmax(node.left.predict_single(row)) for row in X])
            err_left = np.mean(y != left_predictions)
        
        if node.right is not None:
            right_predictions = np.array([np.argmax(node.right.predict_single(row)) for row in X])
            err_right = np.mean(y != right_predictions)
        
        unique_classes, counts = np.unique(y, return_counts=True)
        most_freq_class = unique_classes[np.argmax(counts)]
        err_base = np.mean(y != most_freq_class)
        
        return err_curr, err_left, err_right, err_base

    errors = compute_errors(node, X, y)
    min_err_idx = np.argmin(errors)
    
    # Обрезка в зависимости от минимальной ошибки
    if min_err_idx == 1 and node.left is not None:
        node = node.left
    elif min_err_idx == 2 and node.right is not None:
        node = node.right
    elif min_err_idx == 3:
        node.left = None
        node.right = None
        unique_classes, counts = np.unique(y, return_counts=True)
        most_freq_class = unique_classes[np.argmax(counts)]
        node.prob = np.array([1 if c == most_freq_class else 0 for c in node.classes])
    
    # Рекурсивная обрезка дочерних узлов
    if node.left is not None or node.right is not None:
        mask_left = X[:, node.feature_idx] <= node.predicat
        mask_right = ~mask_left
        
        if node.left is not None:
            X_left, y_left = X[mask_left], y[mask_left]
            node.left = prune_classification_tree(node.left, X_left, y_left)
        
        if node.right is not None:
            X_right, y_right = X[mask_right], y[mask_right]
            node.right = prune_classification_tree(node.right, X_right, y_right)
    
    return node
```
## Результаты
**Деревья:**

По критерию Донского (глубина 8)
![Сравнение регрессии](images/donskoy.png)
По критерию энтропии (глубина 7)
![Сравнение регрессии](images/entropy.png)
По критерию энтропии с прунингом (глубина 7)
![Сравнение регрессии](images/pruning.png)

### Метрики классификации

Результаты работы эталонной модели (`scikit-learn`) и кастомных моделей (с критериями Донского и Энтропии) до и после обрезки дерева:

| Модель                         | Accuracy | Precision | Recall | F1-score |
|--------------------------------|----------|-----------|--------|----------|
| Эталонная (scikit-learn)       | 0.8675   | 0.87      | 0.87   | 0.87     |
| Кастомная модель (Донской)     | 0.8600   | 0.86      | 0.85   | 0.85     |
| Кастомная модель (Энтропия)    | 0.8400   | 0.84      | 0.84   | 0.84     |
| Кастомная после обрезки        | 0.8400   | 0.84      | 0.84   | 0.84     |

### Метрики регрессии

Для задачи регрессии была построена визуализация, сравнивающая истинные значения целевой переменной с предсказаниями эталонной и кастомной моделей:

![Сравнение регрессии](images/regression.png)

### Выводы

1. **Классификация**:
   - Эталонная модель показывает чуть более высокую точность, чем кастомные модели.
   - После обрезки дерева на основе энтропийного критерия качество модели не изменилось, скорее всего, это произошло из-за способа постороения дерева, при котором проход происходит по всем признакам и по всем возможным значениям. На простом датасете это приводит дерево к оптимальному без обрезки.

2. **Регрессия**:
   - Оба подхода (кастомный и эталонный) дают схожие предсказания.
   - Среднеквадратичная ошибка (MSE) у кастомной модели немного выше, чем у `scikit-learn`, что ожидаемо, учитывая более упрощённую логику кастомной реализации.

3. **Время выполнения**:
   - Кастомная реализация значительно медленнее эталонной, особенно на этапах построения дерева и предсказания.
