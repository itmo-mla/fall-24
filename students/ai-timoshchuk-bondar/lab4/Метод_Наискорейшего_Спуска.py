import numpy as np


def main_f(x, y):
    return x * y + 50 / x + 20 / y


def grad_f(x: float, y: float):
    return np.array([y - 50 / x**2, x - 20 / y**2])


def lenght(grad: np.array):
    return np.sqrt(np.sum(grad**2))


def norm(grad: np.array):
    return grad / lenght(grad)


def f(lambd, gradx, grady, x1, y1):
    x = x1 + lambd * gradx
    y = y1 + lambd * grady
    return x * y + 50 / x + 20 / y


def binomial(point:np.array, eps: float):
    a = -1
    b = 1
    while np.abs(a - b) >= 2 * eps:
        # print(f"{a=}, {b=}")
        # print(
        #     f"f(x1): {f((a+b-eps)/2, *norm(grad_f(*point)), *point):3f}, f(x2): {f((a+b+eps)/2, *norm(grad_f(*point)), *point):3f}, \nx1: {(a+b-eps)/2:3f}, x2: {(a+b+eps)/2:3f}"
        # )
        a, b = (
            [(a + b - eps) / 2, b]
            if f((a + b - eps) / 2, *norm(grad_f(*point)), *point)
            > f((a + b + eps) / 2, *norm(grad_f(*point)), *point)
            else [a, (a + b + eps) / 2]
        )
    print(lenght(grad_f(*point)))
    return (a + b) / 2 #, f((a + b) / 2, *norm(grad_f(x, y)), x, y)


# Тут должен быть код, который идёт по тому алгоритму, применяте биномиальный для нахождения оптимального лямбды, и потом делает шаг градиентного метода

point = np.array([1, 1], dtype=np.float64)
eps = 0.00003
while lenght(grad_f(*point))> eps:
    lambd = binomial(point, eps=0.00003)
    print(f"lambd: {lambd}")
    point +=  norm(grad_f(*point)) * lambd
    print(f"x, y == {point}")
