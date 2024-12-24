import numpy as np

def prune_classification_tree(node, X, y):
    """
    Обрезка дерева решений для задачи классификации с целью минимизации ошибок.

    Args:
        node: Узел дерева решений (объект класса Node).
        X (numpy.ndarray): Признаки.
        y (numpy.ndarray): Целевая переменная.

    Returns:
        Узел после возможной обрезки.
    """
    def compute_errors(node, X, y):
        """
        Вычисляет ошибки для текущего узла и возможных упрощений (обрезка до листа, оставление только одного поддерева).
        
        Args:
            node: Узел дерева решений.
            X (numpy.ndarray): Признаки.
            y (numpy.ndarray): Целевая переменная.

        Returns:
            tuple: Ошибки для текущего узла (err_curr), только левого поддерева (err_left),
                   только правого поддерева (err_right) и превращения в лист (err_base).
        """
        # Если узел листовой
        if node.left is None and node.right is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            most_freq_class = unique_classes[np.argmax(counts)]
            predicted_class = np.argmax(node.prob)
            return (
                np.mean(y != predicted_class),
                float('inf'),
                float('inf'),
                np.mean(y != most_freq_class)
            )

        # Ошибка текущего дерева
        predictions = np.array([np.argmax(node.predict_single(row)) for row in X])
        err_curr = np.mean(y != predictions)

        # Сценарий: оставить только левое поддерево
        if node.left is not None:
            left_predictions = np.array([np.argmax(node.left.predict_single(row)) for row in X])
            err_left = np.mean(y != left_predictions)
        else:
            err_left = float('inf')

        # Сценарий: оставить только правое поддерево
        if node.right is not None:
            right_predictions = np.array([np.argmax(node.right.predict_single(row)) for row in X])
            err_right = np.mean(y != right_predictions)
        else:
            err_right = float('inf')

        # Сценарий: превратить узел в лист (выбрать наиболее частый класс)
        unique_classes, counts = np.unique(y, return_counts=True)
        most_freq_class = unique_classes[np.argmax(counts)]
        err_base = np.mean(y != most_freq_class)

        # print(f'err_curr {err_curr}, err_left {err_left}, err_right {err_right}, err_base {err_base}')
        return err_curr, err_left, err_right, err_base

    # Вычисляем ошибки для текущего узла
    errors = compute_errors(node, X, y)
    min_err_idx = np.argmin(errors)
    print(f'Минимальная ошибка по индексу - {min_err_idx}: {errors[min_err_idx]}')
    # Применяем соответствующее упрощение в зависимости от минимальной ошибки
    if min_err_idx == 0:
        # Сохраняем текущий узел
        pass
    elif min_err_idx == 1 and node.left is not None:
        # Заменяем на левое поддерево
        node = node.left
    elif min_err_idx == 2 and node.right is not None:
        # Заменяем на правое поддерево
        node = node.right
    else:
        # Превращаем узел в лист с наиболее частым классом
        unique_classes, counts = np.unique(y, return_counts=True)
        most_freq_class = unique_classes[np.argmax(counts)]
        node.left = None
        node.right = None
        node.prob = np.array([1 if c == most_freq_class else 0 for c in node.classes])

    # Если узел не стал листом, рекурсивно обрезаем дочерние узлы
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
