import numpy as np

h = lambda z: -z * np.log2(z, out=np.zeros_like(z, dtype=np.float64), where=(z!=0))

def donskoy_criteria(X, y):
    """
    Реализация критерия Донского для выбора лучшего порога разделения признака.

    Args:
        X (numpy.ndarray): Массив значений признака (одномерный).
        y (numpy.ndarray): Массив классов (одномерный).

    Returns:
        tuple (float, float): Максимальный прирост информации (max_info_gain) и лучший порог разделения (best_weight).
    """
    max_info_gain = -1  # Максимальный прирост информации
    best_weight = None  # Лучший порог разделения

    # Перебираем уникальные значения признака (кроме последнего, чтобы не разбираться с пустыми группами)
    for predicat in sorted(np.unique(X))[:-1]:
        # Определяем разделение: True, если значение признака > порога
        p = X > predicat

        # Вычисляем прирост информации
        info_gain = np.sum((p[:, None] != p) & (y[:, None] != y))

        # Сохраняем порог, если прирост информации больше текущего максимума
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 4)
            best_weight = predicat

    # Возвращаем наилучший прирост информации и соответствующий порог
    return max_info_gain, best_weight


def h(z):
    """
    Функция для вычисления энтропии, избегая ошибок при нулевых значениях.
    """
    return -z * np.log2(z, out=np.zeros_like(z, dtype=np.float64), where=(z != 0))


def multiclass_entropy_criterion(X, y):
    """
    Критерий энтропии для многоклассового разбиения.

    Args:
        X (numpy.ndarray): Массив значений признака (одномерный).
        y (numpy.ndarray): Массив классов (одномерный).

    Returns:
        tuple (float, float): Максимальный прирост информации (max_info_gain) и лучший порог разделения (best_weight).
    """
    max_info_gain = -1  # Максимальный прирост информации
    best_weight = None  # Лучший порог разделения

    l = len(X)  # Общее число объектов
    unique_classes = np.unique(y)  # Уникальные классы
    p_cls_total = np.array([np.sum(y == cls) for cls in unique_classes])  # Частоты классов

    # Перебираем все возможные пороги разделения, кроме последнего
    for predicat_weight in sorted(np.unique(X))[:-1]:
        # Количество элементов, для которых признак больше текущего порога
        p = np.sum(X > predicat_weight)

        # Частоты классов для правой ветви
        p_cls_totalls_right = np.array([np.sum((y == cls) & (X > predicat_weight)) for cls in unique_classes])

        # Вычисляем энтропию до и после разбиения
        total_entropy = np.sum(h(p_cls_total / l))
        right_entropy = np.sum(h(p_cls_totalls_right / p)) if p > 0 else 0
        left_entropy = np.sum(h((p_cls_total - p_cls_totalls_right) / (l - p))) if l - p > 0 else 0

        # Прирост информации
        info_gain = total_entropy - (p / l * right_entropy) - ((l - p) / l * left_entropy)

        # Сохраняем текущий лучший порог и прирост информации
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 4)
            best_weight = predicat_weight

    return max_info_gain, best_weight


def uncertainty_measure(Y):
    """
    Функция для вычисления меры неопределенности на основе среднеквадратичного отклонения.

    Args:
        Y (numpy.ndarray): Массив значений целевой переменной.

    Returns:
        float: Мера неопределенности.
    """
    mean_y = np.mean(Y)  # Среднее значение
    return np.mean((Y - mean_y) ** 2)  # Среднеквадратичное отклонение


def mse_criteria(X, y):
    """
    Критерий MSE (Mean Squared Error) для выбора лучшего порога разделения.

    Args:
        X (numpy.ndarray): Массив значений признака (одномерный).
        y (numpy.ndarray): Массив целевых переменных (одномерный).

    Returns:
        tuple (float, float): Максимальный прирост информации (max_info_gain) и лучший порог разделения (best_weight).
    """
    max_info_gain = -1  # Максимальный прирост информации
    best_weight = None  # Лучший порог разделения

    l = len(y)  # Общее количество объектов

    # Перебираем все уникальные пороги разделения, кроме последнего
    for predicat_weight in sorted(np.unique(X))[:-1]:
        # Логическое условие для разделения данных
        mask_right = X > predicat_weight
        mask_left = ~mask_right

        # Разделение y на правую и левую части
        y_right = y[mask_right]
        y_left = y[mask_left]

        # Вычисляем прирост информации
        total_uncertainty = uncertainty_measure(y)
        right_uncertainty = uncertainty_measure(y_right) if len(y_right) > 0 else 0
        left_uncertainty = uncertainty_measure(y_left) if len(y_left) > 0 else 0

        info_gain = (total_uncertainty 
                     - (len(y_right) / l) * right_uncertainty 
                     - (len(y_left) / l) * left_uncertainty)

        # Сохраняем текущий лучший порог и прирост информации
        if info_gain > max_info_gain:
            max_info_gain = round(info_gain, 5)
            best_weight = predicat_weight

    return max_info_gain, best_weight
