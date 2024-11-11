import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from functools import partial

# Функция для рисования эллипсов
def draw_ellipse(mu, sigma, ax, color):
    width = 2 * np.sqrt(sigma[0])  # ширина эллипса (по оси x)
    height = 2 * np.sqrt(sigma[1])  # высота эллипса (по оси y)
    ell = Ellipse(xy=mu, width=width, height=height, edgecolor=color, fc='None', lw=2)
    ax.add_patch(ell)

# Функция для обновления графика на каждой итерации
def update(frame, ax, points, iterations_data):
    ax.clear()  # Очищаем предыдущий график
    
    # Извлекаем данные для текущей итерации
    data = iterations_data[frame]
    g = data['g']
    mu_y = data['mu_y']
    sigma_y = data['sigma_y']
    
    # Рисуем точки данных
    # print("SHAPE", points.shape[0])
    # for i in range(points.shape[1]):
        # print(i)

    sizes = g * 1000  # Масштабируем для лучшей видимости
    colors = ['red', 'blue', 'green', 'orange', 'black', 'yellow', 'brown'][:g.shape[0]]
    # ax.scatter(points[0, , 0], points[0, i, 1], s=sizes[j], color=colors[j], alpha=0.6)
    for j in range(g.shape[0]):
        ax.scatter(points[0, :, 0], points[0, :, 1], s=sizes[j, :], color=colors[j], alpha=0.6)
    
    # Рисуем центры кластеров и эллипсы ковариаций
    for i in range(mu_y.shape[0]):
        ax.scatter(mu_y[i, 0, 0], mu_y[i, 0, 1], color='black', marker='x', s=200)
        draw_ellipse(mu_y[i, 0], sigma_y[i, 0], ax, color=colors[i])
    
    # Оформление графика
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'EM Clustering - Iteration {frame+1}')
    ax.grid(True)

def make_gif(points, iterations_data):
    # Настройка фигуры и осей для графика
    fig, ax = plt.subplots()

    # Создаем анимацию с помощью FuncAnimation, передавая ax через partial
    ani = FuncAnimation(fig, partial(update, ax=ax, points=points, iterations_data=iterations_data), 
                        frames=len(iterations_data), repeat=False)

    # Сохранение анимации как GIF
    writer = PillowWriter(fps=1)  # Количество кадров в секунду
    ani.save("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/em_clustering.gif", writer=writer)

    plt.show()

# make_gif(points=points, iterations_data=iterations_data, mu=mu_y, sigma=sigma_y)    
    
    
    
    #ani.save("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/em_clustering.gif", writer=writer)