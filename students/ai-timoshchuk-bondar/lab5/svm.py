import numpy as np


class My_SVM:
    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1,
        poly_deg: float = 3,
        rad_gamma: float = 1,
        sigm_gamma: float = 13,
        sigm_betta: float = 0,
        max_iter: int = 1000,
    ):
        # Сразу при инициализации выбирается ядро, и его функция записывается в self.kernel
        self.kernel_transform = {
            "linear": lambda x_i, x_j: np.dot(x_i, x_j.T),
            "poly": lambda x_i, x_j: (1 + np.dot(x_i, x_j.T)) ** poly_deg,
            "rad_func": lambda x_i, x_j: np.exp(
                -rad_gamma * np.sum((x_j - x_i[:, np.newaxis]) ** 2, axis=-1)
            ),
            "sigm": lambda x_i, x_j: np.tanh(
                sigm_gamma * np.dot(x_i, x_j.T) + sigm_betta
            ),
        }[kernel]

        self.C = C
        self.max_iter = max_iter

    def check_max_c(self, v, t, u):
        # Здесь отсекаем выход за пределы для нулевого  индекса

        t_zer_cheked = (np.clip(a=v + t * u, a_min=0, a_max=self.C) - v)[1] / u[
            1
        ]  # .reshape(-1, 1)
        return (np.clip(a=v + t_zer_cheked * u, a_min=0, a_max=self.C) - v)[0] / u[0]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # Магическая (не совсем) штука, которая делает deepcopy без использования deepcopy
        # Тут прикол в том, что умножение создаёт новую ссылку на объект, если для него не указано другое.
        # Сответственно, так как это не листы, то мы переписываем ссылки для всех объектов вглубь,
        # По сути делая тот же самый deepcopy только средствами numpy
        self.X = X * 1
        self.y = Y * 2 - 1
        # print(self.y)
        self.lambdas = np.zeros_like(self.y, dtype=np.float32)
        self.Kyiyj = (
            self.kernel_transform(self.X, self.X) * self.y[:, np.newaxis] * self.y
        )

        for _ in range(self.max_iter):
            for ind_M in range(self.lambdas.shape[0]):

                # пока выбирается рандомно, потом можно переделать на более интересный выбор
                ind_L = np.random.randint(0, self.lambdas.shape[0])


                Q = self.Kyiyj[
                    [[ind_M, ind_M], [ind_M, ind_L]], [[ind_L, ind_M], [ind_L, ind_L]]
                ]
                v_0_T = self.lambdas[[ind_M, ind_L]]
                v_0 = v_0_T.reshape(-1, 1)
                k_0_T = 1 - self.lambdas @ self.Kyiyj[[ind_M, ind_L]].T

                u = np.array([[-self.y[ind_L]], [self.y[ind_M]]])

                t_zvezda = (k_0_T @ u) / ((u.T @ Q @ u) + 1e-15)

                t_zvezda = self.check_max_c(v_0, t_zvezda, u)

                self.lambdas[[ind_M, ind_L]] = v_0_T + t_zvezda * u.T  # .reshape(2)


        (self.opor_idx,) = np.nonzero(self.lambdas > 1e-10)


        self.b = np.mean(
            (1.0 - np.sum(self.Kyiyj[self.opor_idx] * self.lambdas, axis=1))
            * self.y[self.opor_idx]
        )
        print(f" start {self.b=}")

        print(f"{self.opor_idx=}")

        print(f"{np.sum(self.lambdas*self.y)=}")

        print(
            f"{np.mean(self.y[self.opor_idx] - np.sum(self.y[self.opor_idx] * self.lambdas[self.opor_idx] * self.kernel_transform(self.X[self.opor_idx], self.X[self.opor_idx]), axis=0))=}"
        )
        self.b = np.mean(
            self.y[self.opor_idx]
            - np.sum(
                self.lambdas[self.opor_idx]
                * self.y[self.opor_idx]
                * self.kernel_transform(self.X[self.opor_idx], self.X[self.opor_idx]),
                axis=0,
            )
        )
        print(f"stop {self.b=}")

    def decision_function(self, X: np.array):

        return (
            np.sum(
                self.kernel_transform(X, self.X[self.opor_idx])
                * self.y[self.opor_idx]
                * self.lambdas[self.opor_idx],
                axis=1,
            )
            + self.b
        )

    def predict(self, X: np.array):
        return (np.sign(self.decision_function(X)) + 1) // 2


svm_my = My_SVM(max_iter=60, kernel="linear")
X = np.array([[1, 2], [3, 4], [3, 9], [-1, -2], [-3, -4], [-4, -2], [-1, -1], [1, 8]])
y = np.array([1, 1, 1, 0, 0, 0, 0, 1])


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
test_plot(X, y, svm_my, axs[0], "MySVM")
svm_my.predict(np.array([[1, 2], [-3, -4]]))
