# Метод опорных векторов (SVM)

## 1. Основные понятия

### Задача обучения линейного классификатора

**Дано:**
- Обучающая выборка $X_l = \{(x_i, y_i)\}_{i=1}^l$
- $x_i$ - объекты, векторы из множества $X = \mathbb{R}^n$
- $y_i$ - метки классов из множества $Y = \{-1, +1\}$

**Найти:**
- Параметры $w \in \mathbb{R}^n$, $w_0 \in \mathbb{R}$ линейной модели классификации:

$a(x;w,w_0) = \text{sign}(\langle x,w \rangle - w_0)$

**Критерий** - минимизация эмпирического риска:

$\sum_{i=1}^l [a(x_i;w,w_0) \neq y_i] = \sum_{i=1}^l [M_i(w,w_0) < 0] \to \min_{w,w_0}$

где $M_i(w,w_0) = (\langle x_i,w \rangle - w_0)y_i$ - отступ (margin) объекта $x_i$

### Оптимальная разделяющая гиперплоскость

Для линейно разделимой выборки существуют $w,w_0$ такие, что:

$M_i(w,w_0) = y_i(\langle w, x_i \rangle - w_0) > 0, i = 1,\ldots,l$

При нормировке $\min_{i=1,\ldots,l} M_i(w,w_0) = 1$ получаем разделяющую полосу:

$\{x: -1 \leq \langle w, x \rangle - w_0 \leq 1\}$

Ширина полосы равна $\frac{2}{\|w\|}$, что приводит к задаче максимизации зазора. 
Геометрически это следует из того, что вектор $w$ перпендикулярен разделяющей гиперплоскости 
(так как градиент $\nabla_x(\langle w,x \rangle - w_0) = w$ задает направление наибольшего изменения значения функции, 
которое всегда перпендикулярно линиям уровня, в том числе и нулевой линии уровня - разделяющей гиперплоскости), 
а расстояние от точки до гиперплоскости вычисляется как $\frac{|\langle w,x \rangle - w_0|}{\|w\|}$. 
Поскольку границы полосы заданы уравнениями $\langle w,x \rangle - w_0 = \pm 1$, 
расстояние между ними равно $\frac{2}{\|w\|}$.

## 2. Постановка задачи оптимизации

### Задача с мягкими границами

В большинстве случаев выборка линейно неразделима, поэтому вводится дополнительные переменные $\xi_i$, которые позволяют некоторым точкам нарушать границы разделяющей полосы.

В этой формулировке:
- $w$ - вектор весов, определяющий направление разделяющей гиперплоскости
- $w_0$ - порог (смещение гиперплоскости)
- $\xi_i$ - переменные ошибок (slack variables), позволяющие некоторым точкам нарушать границы разделяющей полосы
- $C$ - параметр регуляризации, контролирующий баланс между максимизацией ширины разделяющей полосы и минимизацией ошибок

$\begin{cases}
\frac{1}{2}\|w\|^2 + C\sum_{i=1}^l \xi_i \to \min_{w,w_0,\xi} \\
\xi_i \geq 1 - M_i(w,w_0), i = 1,\ldots,l \\
\xi_i \geq 0, i = 1,\ldots,l
\end{cases}$

Первое слагаемое $\frac{1}{2}\|w\|^2$ максимизирует ширину разделяющей полосы, второе $C\sum_{i=1}^l \xi_i$ штрафует за ошибки классификации.

Эквивалентная задача безусловной минимизации использует функцию потерь хинджа $(1 - M_i(w,w_0))_+$:

$C\sum_{i=1}^l (1 - M_i(w,w_0))_+ + \frac{1}{2}\|w\|^2 \to \min_{w,w_0}$

### Двойственная задача


#### Теорема Куна-Таккера

Теорема формулирует необходимые условия минимума функции при наличии ограничений.

Пусть $\hat{x} \in \arg \min f$ при наложенных ограничениях является решением задачи. Тогда существует вектор множителей Лагранжа $\lambda \in \mathbb{R}^m$ такой, что для функции Лагранжа:

$L(x) = \lambda_0f(x) + \sum_{i=1}^m \lambda_ig_i(x)$

выполняются следующие условия:

1. Условие стационарности:

   $\min_x L(x) = L(\hat{x})$

2. Условие дополняющей нежёсткости:

   $\lambda_ig_i(\hat{x}) = 0, i = 1,\ldots,m$

3. Условие неотрицательности множителей Лагранжа:

   $\lambda_i \geq 0, i = 1,\ldots,m$

Эти условия являются необходимыми для существования минимума целевой функции при заданных ограничениях.

#### Применение теоремы Куна-Таккера

Исходная задача с мягкими границами:

$\begin{cases}
\frac{1}{2}\|w\|^2 + C\sum_{i=1}^l \xi_i \to \min_{w,w_0,\xi} \\
y_i(\langle w,x_i \rangle - w_0) \geq 1 - \xi_i, i = 1,\ldots,l \\
\xi_i \geq 0, i = 1,\ldots,l
\end{cases}$

Функция Лагранжа для этой задачи:

$L(w,w_0,\xi,\lambda,\mu) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^l \xi_i - \sum_{i=1}^l \lambda_i[y_i(\langle w,x_i \rangle - w_0) - (1 - \xi_i)] - \sum_{i=1}^l \mu_i\xi_i$

где:
- $\lambda_i \geq 0$ - переменные, двойственные к ограничениям $M_i(w,w_0) \geq 1 - \xi_i$
- $\mu_i \geq 0$ - переменные, двойственные к ограничениям $\xi_i \geq 0$

#### Условия Куна-Таккера

1. Условия стационарности:

   - $\frac{\partial L}{\partial w} = w - \sum_{i=1}^l \lambda_iy_ix_i = 0$
   - $\frac{\partial L}{\partial w_0} = \sum_{i=1}^l \lambda_iy_i = 0$
   - $\frac{\partial L}{\partial \xi_i} = C - \lambda_i - \mu_i = 0$

2. Условия дополняющей нежесткости:

   - $\lambda_i[y_i(\langle w,x_i \rangle - w_0) - (1 - \xi_i)] = 0$
   - $\mu_i\xi_i = 0$

3. Двойственные допустимости:

   - $\lambda_i \geq 0$
   - $\mu_i \geq 0$

#### Вывод двойственной задачи

1. Из условия стационарности по $w$:

   $w = \sum_{i=1}^l \lambda_iy_ix_i$

2. Из условий по $\xi_i$:

   $C = \lambda_i + \mu_i$ и $\mu_i \geq 0$ ⟹ $0 \leq \lambda_i \leq C$

3. Подставляя эти выражения в функцию Лагранжа и упрощая:

   $L(\lambda) = \sum_{i=1}^l \lambda_i - \frac{1}{2}\sum_{i=1}^l\sum_{j=1}^l \lambda_i\lambda_jy_iy_j\langle x_i, x_j\rangle$

4. Для нахождения порога $b$ используем условие дополняющей нежесткости:

   $\lambda_i[y_i(\langle w,x_i \rangle - b) - (1 - \xi_i)] = 0$

   Для опорных векторов, где $0 < \lambda_i < C$ (и следовательно $\xi_i = 0$):
   
   $y_i(\langle w,x_i \rangle - b) = 1$
   
   Подставляя выражение для $w$:
   
   $y_i(\sum_{j=1}^l \lambda_jy_j\langle x_j,x_i \rangle - b) = 1$
   
   Отсюда:
   
   $b = \sum_{j=1}^l \lambda_jy_j\langle x_j,x_i \rangle - y_i$

   На практике для численной устойчивости часто берут среднее значение по всем опорным векторам на границе полосы:

   $b = \frac{1}{|S|}\sum_{i \in S} \left(\sum_{j=1}^l \lambda_jy_j\langle x_j,x_i \rangle - y_i\right)$

   где $S = \{i: 0 < \lambda_i < C\}$ - множество индексов опорных векторов на границе полосы.

Итоговая двойственная задача:

$\begin{cases}
-L(\lambda) = -\sum_{i=1}^l \lambda_i + \frac{1}{2}\sum_{i=1}^l\sum_{j=1}^l \lambda_i\lambda_jy_iy_j\langle x_i, x_j\rangle \to \min_{\lambda} \\
\sum_{i=1}^l \lambda_i y_i = 0 \\
0 \leq \lambda_i \leq C, i = 1,\ldots,l
\end{cases}$

#### Интерпретация решения

- Объекты с $\lambda_i > 0$ являются опорными векторами
- Если $0 < \lambda_i < C$, то объект лежит точно на границе разделяющей полосы
- Если $\lambda_i = C$, то объект либо внутри полосы, либо классифицирован неверно
- Вектор весов восстанавливается как $w = \sum_{i=1}^l \lambda_iy_ix_i$
- Порог $w_0$ находится из условий дополняющей нежесткости

Типизация объектов $x_i$, i = 1, . . . , l:

1. $\lambda_i = 0; \eta_i = C; \xi_i = 0; M_i \geq 1$ — периферийный.
2. $0 < \lambda_i < C; 0 < \eta_i < C; \xi_i = 0; M_i = 1$ — опорный-граничный.
3. $\lambda_i = C; \eta_i = 0; \xi_i > 0; M_i < 1$ — опорный-нарушитель.

## 3. Нелинейное обобщение SVM

### Ядра

Идея: заменить скалярное произведение $\langle x,x' \rangle$ на нелинейную функцию $K(x,x')$

#### Теоретическое обоснование

Согласно теореме Мерсера, любое положительно определенное ядро $K(x,x')$ может быть представлено как скалярное произведение в некотором расширенном пространстве признаков $\mathcal{H}$:

$K(x,x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$

где $\phi: X \to \mathcal{H}$ - некоторое отображение исходного пространства в пространство признаков (возможно бесконечномерное).

Функция $K: X \times X \to \mathbb{R}$ является ядром, если:
1. Симметрична: $K(x,x') = K(x',x)$
2. Неотрицательно определена: $\iint_X K(x,x')g(x)g(x') dx dx' \geq 0$ для любой $g: X \to \mathbb{R}$

#### Преимущества использования ядер

1. **Неявное преобразование признаков**: Нет необходимости явно вычислять отображение $\phi(x)$, которое может быть очень сложным или бесконечномерным
2. **Эффективность вычислений**: Вместо работы в пространстве высокой размерности, все вычисления производятся через функцию ядра в исходном пространстве ("kernel trick")
3. **Нелинейная классификация**: Позволяет строить нелинейные разделяющие поверхности в исходном пространстве, оставаясь в рамках линейной оптимизации
4. **Универсальность**: Для некоторых ядер (например, RBF) доказана возможность аппроксимации любой непрерывной функции с любой точностью

### Популярные ядра

1. Полиномиальное: $K(x,x') = (\langle x,x' \rangle + 1)^d$
   - Соответствует всем возможным мономам степени не выше d
   - Параметр d определяет сложность модели

2. RBF (гауссовское): $K(x,x') = \exp(-\gamma\|x-x'\|^2)$
   - Соответствует бесконечномерному пространству признаков
   - Параметр $\gamma$ управляет локальностью влияния опорных векторов
   - Наиболее популярное ядро на практике

3. Сигмоидное: $K(x,x') = \tanh(k_1\langle x,x' \rangle - k_0)$
   - Связь с нейронными сетями: эквивалентно двухслойной нейронной сети
   - Не всегда удовлетворяет условиям Мерсера

## 4. Преимущества и недостатки SVM

### Преимущества:
- Единственное решение задачи выпуклой оптимизации
- Автоматическое определение числа опорных векторов
- Эффективная работа в пространствах высокой размерности
- Гибкость за счет выбора ядра

### Недостатки:
- Отсутствие общих подходов к выбору ядра
- Медленное обучение на больших данных
- Необходимость подбора параметра регуляризации C
- Отсутствие встроенного механизма отбора признаков

## Решение задачи

### Набор данных

https://www.kaggle.com/datasets/abrambeyer/openintro-possum

Набор данных по опоссумам состоит из девяти морфометрических измерений, сделанных на 104 горных щеткохвостых опоссумах, пойманных в семи местах от Южной Виктории до центрального Квинсленда.

| site | Pop | sex | age | hdlngth | skullw | totlngth | taill | footlgth | earconch | eye | chest | belly |
|------|-----|-----|-----|---------|---------|-----------|--------|-----------|-----------|-----|--------|--------|
| 1 | Vic | m | 8.0 | 94.1 | 60.4 | 89.0 | 36.0 | 74.5 | 54.5 | 15.2 | 28.0 | 36.0 |
| 1 | Vic | f | 6.0 | 92.5 | 57.6 | 91.5 | 36.5 | 72.5 | 51.2 | 16.0 | 28.5 | 33.0 |
| 1 | Vic | f | 6.0 | 94.0 | 60.0 | 95.5 | 39.0 | 75.4 | 51.9 | 15.5 | 30.0 | 34.0 |
| 1 | Vic | f | 6.0 | 93.2 | 57.1 | 92.0 | 38.0 | 76.1 | 52.2 | 15.2 | 28.0 | 34.0 |
| 1 | Vic | f | 2.0 | 91.5 | 56.3 | 85.5 | 36.0 | 71.0 | 53.2 | 15.1 | 28.5 | 33.0 |

### Реализация на python

```python
from typing import Protocol
import numpy as np
import scipy.optimize


class SVMKernel(Protocol):
    """Protocol defining the interface for SVM kernel functions"""

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function value for two vectors"""
        ...


class LinearKernel:
    """Linear kernel: K(x,x') = <x,x'>"""

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)

    def __str__(self) -> str:
        return "Linear"


class RBFKernel:
    """RBF (Gaussian) kernel: K(x,x') = exp(-gamma||x-x'||^2)"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def __str__(self) -> str:
        return f"RBF(gamma={self.gamma})"


class PolynomialKernel:
    """Polynomial kernel: K(x,x') = (1 + <x,x'>)^degree"""

    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return (1 + np.dot(x1, x2)) ** self.degree

    def __str__(self) -> str:
        return f"Polynomial(degree={self.degree})"


class SVM:
    def __init__(self, kernel: SVMKernel, C: float = 1.0):
        """
        Initialize SVM classifier

        Args:
            kernel: Kernel function to use
            C: Regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.lambdas: np.ndarray | None = None  # Lagrange multipliers
        self.support_vectors: np.ndarray | None = None
        self.support_vector_labels: np.ndarray | None = None
        self.b: float | None = None  # Intercept term

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix for given data"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # Compute diagonal element
            K[i, i] = self.kernel(X[i], X[i])
            # Compute upper triangle and mirror to lower triangle
            for j in range(i + 1, n_samples):
                K[i, j] = self.kernel(X[i], X[j])
                K[j, i] = K[i, j]  # Kernel is symmetric
        return K

    def _objective(self, lambdas: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the dual objective function to minimize:
        -L(λ) = -Σλᵢ + (1/2)ΣΣλᵢλⱼyᵢyⱼK(xᵢ,xⱼ)
        """
        return -np.sum(lambdas) + 0.5 * np.sum(np.outer(lambdas * y, lambdas * y) * K)

    def _objective_gradient(self, lambdas: np.ndarray, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of the dual objective function"""
        return -np.ones_like(lambdas) + np.sum(lambdas * y * K.T, axis=1) * y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit the SVM classifier

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be [-1, 1]")

        n_samples = X.shape[0]

        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == -1)

        # Validate if the problem is solvable
        if n_positive == 0 or n_negative == 0:
            raise ValueError("Cannot solve SVM with only one class")

        # if self.C * min(n_positive, n_negative) < max(n_positive, n_negative):
        # print("Warning: C might be too small for the class imbalance")

        K = self._compute_kernel_matrix(X)

        # Modified optimization setup with looser tolerances
        solution = scipy.optimize.minimize(
            fun=lambda x: self._objective(x, K, y),
            x0=np.zeros(n_samples),
            method="SLSQP",  # Sequential Least Squares Programming
            jac=lambda x: self._objective_gradient(x, K, y),
            bounds=[(0, self.C) for _ in range(n_samples)],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x, y=y: np.sum(x * y),
                }
            ],
            options={
                'maxiter': 1000,
                'ftol': 1e-6,
                'disp': False
            }
        )

        if not solution.success:
            raise ValueError(f"Optimization did not converge: {solution.message}")

        # Get the optimal Lagrange multipliers
        self.lambdas = solution.x

        # Find support vectors (points with λᵢ > 0)
        sv_threshold = 1e-5  # Numerical tolerance
        sv_idx = self.lambdas > sv_threshold

        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y[sv_idx]
        self.lambdas = self.lambdas[sv_idx]

        if len(self.support_vectors) == 0:
            raise ValueError("No support vectors found. Model may not be useful.")

        # Compute intercept term b
        self.b = self._compute_intercept()

        return self

    def _compute_intercept(self) -> float:
        """
        Compute the intercept term using support vectors
        that lie on the margin (0 < λᵢ < C)
        """

        if len(self.support_vectors) == 0:
            return 0.0

        margin_threshold = 1e-5
        margin_idx = (self.lambdas > margin_threshold) & (
            self.lambdas < self.C - margin_threshold
        )

        if np.any(margin_idx):
            # Use first support vector on margin to compute b
            sv = self.support_vectors[margin_idx][0]
            sv_y = self.support_vector_labels[margin_idx][0]

            # b = yᵢ - Σλⱼyⱼk(xⱼ,xᵢ)
            b = sv_y - np.sum(
                self.lambdas
                * self.support_vector_labels
                * np.array([self.kernel(sv, x) for x in self.support_vectors])
            )
        else:
            # Fallback: use average of predictions
            b = 0
            for i, sv in enumerate(self.support_vectors):
                b += self.support_vector_labels[i] - np.sum(
                    self.lambdas
                    * self.support_vector_labels
                    * np.array([self.kernel(sv, x) for x in self.support_vectors])
                )
            b /= len(self.support_vectors)

        return b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X

        Args:
            X: Data points to classify, shape (n_samples, n_features)

        Returns:
            Predicted class labels (-1 or 1)
        """
        decision_values = (
            np.array(
                [
                    np.sum(
                        self.lambdas
                        * self.support_vector_labels
                        * np.array([self.kernel(x, sv) for sv in self.support_vectors])
                    )
                    for x in X
                ]
            )
            - self.b
        )

        return np.sign(decision_values)

    def get_hyperplane_parameters(self) -> tuple[np.ndarray, float]:
        """
        Get the parameters of the separating hyperplane (w, b)
        Only works for linear kernel.
        
        Returns:
            tuple: (w, b) where w is the normal vector and b is the intercept
        """
        if not isinstance(self.kernel, LinearKernel):
            raise ValueError("Hyperplane parameters can only be computed for linear kernel")
            
        if self.support_vectors is None or self.support_vector_labels is None:
            raise ValueError("Model must be fitted first")
            
        # For linear kernel, w = Σᵢ λᵢyᵢxᵢ
        w = np.sum(self.lambdas[:, np.newaxis] * 
                  self.support_vector_labels[:, np.newaxis] * 
                  self.support_vectors, axis=0)
                  
        return w, self.b
```

### Визуализация решения

![svm-vis](assets/svm_vis.png)

### Сравнение результатов с sklearn

| Kernel               |    C | Implementation   |   Recall |   Precision |   F1 Score |
|:---------------------|-----:|:-----------------|---------:|------------:|-----------:|
| Linear               | 0.01 | Custom           | 0        |    0        |   0        |
| Linear               | 0.1  | Custom           | 0.416667 |    0.714286 |   0.526316 |
| Linear               | 0.5  | Custom           | 0.416667 |    0.714286 |   0.526316 |
| Polynomial(degree=3) | 0.01 | Custom           | 0.916667 |    0.733333 |   0.814815 |
| Polynomial(degree=3) | 0.1  | Custom           | 0.833333 |    0.714286 |   0.769231 |
| Polynomial(degree=3) | 0.5  | Custom           | 0.833333 |    0.714286 |   0.769231 |
| RBF(gamma=1.0)       | 0.01 | Custom           | 0        |    0        |   0        |
| RBF(gamma=1.0)       | 0.1  | Custom           | 0        |    0        |   0        |
| RBF(gamma=1.0)       | 0.5  | Custom           | 0        |    0        |   0        |

Sklearn SVM Results:
| Kernel               |    C | Implementation   |   Recall |   Precision |   F1 Score |
|:---------------------|-----:|:-----------------|---------:|------------:|-----------:|
| Linear               | 0.01 | Sklearn          | 1        |    0.571429 |   0.727273 |
| Linear               | 0.1  | Sklearn          | 0.666667 |    0.8      |   0.727273 |
| Linear               | 0.5  | Sklearn          | 0.666667 |    0.8      |   0.727273 |
| Polynomial(degree=3) | 0.01 | Sklearn          | 1        |    0.571429 |   0.727273 |
| Polynomial(degree=3) | 0.1  | Sklearn          | 1        |    0.571429 |   0.727273 |
| Polynomial(degree=3) | 0.5  | Sklearn          | 0.916667 |    0.578947 |   0.709677 |
| RBF(gamma=1.0)       | 0.01 | Sklearn          | 1        |    0.571429 |   0.727273 |
| RBF(gamma=1.0)       | 0.1  | Sklearn          | 1        |    0.571429 |   0.727273 |
| RBF(gamma=1.0)       | 0.5  | Sklearn          | 1        |    0.571429 |   0.727273 |
