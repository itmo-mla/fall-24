import numpy as np

class LinearClassifier:
    def __init__(self, weights_init_method, reg_coef, lambd, gamma, obj_presentation, epochs=50):
        # Метод инициализации весов
        self.weights_init_method = weights_init_method
        # Коэф. регуляризации
        self.reg_coef = reg_coef
        # Темп забывания предыстории ряда
        self.lambd = lambd
        # Коэф. инерции
        self.gamma = gamma
        # Способ предъявления объектов
        self.obj_presentation = obj_presentation
        self.epochs = epochs
        self.W = None
        self.v = None  

    # Вычисление отступа
    def margin(self, X, y):
        return (X @ self.W) * y
    
    # Среднеквадратичная ошибка
    def loss(self, X, y):
        return np.mean((y - (X @ self.W)) ** 2 + self.reg_coef * np.sum(self.W ** 2) / 2)

    # Вычисление градиента ф. потерь
    def grad(self, X, y, W):
        # Штрафуем за размер весов, добавляя к градиенту регуляризационный член
        return -2 * (X.T @ (y - (X @ W))) / len(y) + self.reg_coef * self.W
    
    # Случайное предъявление объектов
    def random_indices(self, X_train):
        return np.random.choice(X_train.shape[0], 10, replace=False)
    
    # Предъявление объектов по модулю отступа (чем меньше |M|, тем выше вероятность выбора)
    def margin_indices(self, X_train, y_train):
        margins = np.abs(self.margin(X_train, y_train))
        exp_margins = np.exp(-margins)
        p = exp_margins / np.sum(exp_margins)
        return np.random.choice(X_train.shape[0], 10, False, p)

    def fit(self, X_train, y_train):
        # Инициализация весов
        match self.weights_init_method:
            case "correlation":
                weights = np.zeros(X_train.shape[1])
                for i in range(X_train.shape[1]):
                    # Для каждого признака считаем веса, отражающие корреляцию м-у признаками и таргетами
                    # (отношение суммы произв. таргетов на признак к сумме квадратов признака)
                    weights[i] = (y_train @ X_train[:, i]) / ((X_train[:, i] @ X_train[:, i]))
                self.W = weights
            case "multistart":
                min_loss, weights = 1, None
                # Генерим 20 наборов весов  из равномерного распределения
                for i in range(20):
                    self.W = np.random.uniform(-(1 / (2 * X_train.shape[1])), 1 / (2 * X_train.shape[1]), X_train.shape[1])
                    loss = self.loss(X_train, y_train)
                    # И выбираем тот, при котором достигается наименьший лосс
                    if loss < min_loss:
                        weights = self.W 
                        min_loss = loss
                self.W = weights
            case "default":
                # В случае отсутствия предпочтений по весам просто генерируем их из равн-го распределения
                self.W = np.random.uniform(-(1 / (2 * X_train.shape[1])), 1 / (2 * X_train.shape[1]), X_train.shape[1])

        # Определяем способ предъявления объектов
        if self.obj_presentation == 'margin_abs':
            indices = self.margin_indices(X_train, y_train)
        else: 
            indices = self.random_indices(X_train)
        self.v = np.zeros_like(self.W)
        # В начале оценка ф. качества - ср. значение лосса по выборке
        Q = np.mean(self.loss(X_train[indices], y_train[indices]))

        for _ in range(self.epochs):
            if self.obj_presentation == 'margin_abs':
                indices = self.margin_indices(X_train, y_train)
            else: 
                indices = self.random_indices(X_train)
            # Скорейший GD (нормируем шаг по норме выборки)
            lr = 1 / np.linalg.norm(X_train[indices]) ** 2
            # Стохастический градиентный спуск с инерцией
            loss_grad = self.grad(X_train[indices], y_train[indices], W = self.W - lr * self.gamma * self.v) 
            self.v = self.gamma * self.v + (1 - self.gamma) * loss_grad 
            self.W = self.W * (1 - lr * self.reg_coef) - lr * self.v
            # Рекуррентная оценка функционала качества
            ei = self.loss(X_train[indices], y_train[indices]) 
            Q = self.lambd * ei + (1 - self.lambd) * Q

    def predict(self, X):
        return np.sign((X @ self.W))
