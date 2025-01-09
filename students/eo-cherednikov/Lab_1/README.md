# Лабораторная работа №1

# Визуализация данных
### Датасет 1: Iris Clustering
[Ссылка на датасет](https://www.kaggle.com/datasets/uciml/iris)

<img title="Визуализация данных" src="img/iris.png" width="600">

**Гипотеза**: судя по рисунку, алгоритмы должны выделять 3-4 кластера.

### Датасет 2: Mall Customers 
[Ссылка на датасет](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

<img title="Визуализация данных" src="img/mall.png" width="600">

**Гипотеза**: алгоритмы должны выделять 4-5 кластеров. PCA плохо справилcя с визуализацией, у TSNE получились более выделенные результаты. 

# Иерархический алгоритм и дендрограмма
<img title="Дендрограмма: Iris" src="img/dendrogram_Iris.png" width="600">
<img title="Дендрограмма: Customers" src="img/dendrogram_Mall.png" width="600">

Видно, что дендрограммы, построенные разработанным алгоритмом и его реализацией в scipy аналогичны. 

<img title="Сравнение и метрики" src="img/stats_hierarchical_Iris.png" width=400>
<img title="Сравнение и метрики" src="img/stats_hierarchical_Mall.png" width=400>


По графикам видно, что среднее растояние между кластерами и внутри них совпадают. Видна большая разница во времени исполнения, реализованный алгоритм значительно медленнее.

Алгоритм также определил оптимальное число кластеров: для IrisClustering - 4 кластера, для Customers - 5.

# EM-алгоритм
### Iris Clustering
<img title="EM-алгоритм: Iris" src="img/em_Iris.png" width=600>

На рисунке показано распределение меток на построенных данных. Данные были преобразованы к стандартному распределению при помощи `StandardScaler` из библиотеки `sklearn`.

<img title='Сравнение и метрики' src='img/stats_EM-algorithm_Iris.png' width=400>

На рисунке показаны метрики. По времени реализованный алгоритм отрабатывает медленнее, расстояния между кластерами почти совпадают, расстояния между кластерами отличаются сильнее

### Customers
<img title="EM-алгоритм: Customers" src="img/em_Mall.png" width=600>

<img title='Сравнение и метрики' src='img/stats_EM-algorithm_Mall.png' width=400>

<img title="Дополнительное сравнение при TSNE" src="img/additional_em_Mall_native.png">

<img title="Дополнительное сравнение при TSNE" src="img/additional_em_Mall_sklearn.png">

# DBSCAN
### Iris
<img title="DBSCAN: Iris" src="img/dbscan_Iris.png" width=600>

На рисунке показано распределение меток на построенных данных. Здесь `min_samples=15`, `epsilon=1.5`.
### Customers
<img title="DBSCAN: Customers" src="img/dbscan_Mall.png" width=600>

На рисунке показано распределение меток на построенных данных. Здесь `min_samples=15`, `epsilon=1.5`.
