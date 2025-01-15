# LAB 2

Был реализован алгоритм KNN с методом окна Парзена переменной ширины

Для подбора параметра k был применен метод Leave One Out

После реализации алгоритма и подбора параметра k, было проведено сравнение нативной реализации с эталонной

Результаты сравнения: <br>
Optimal k (custom implementation): 1 <br>
Execution time (custom implementation): 114395 microseconds <br>
Accuracy (custom implementation): 98.33% <br>
Precision: 0.9841 <br>
Recall: 0.9833 <br>
F1-score: 0.9833 <br>
Optimal k (library implementation): 5 <br>
Execution time (library implementation): 86710 microseconds <br>
Accuracy (library implementation): 96.67% <br>
Precision: 0.9694 <br>
Recall: 0.9667 <br>
F1-score: 0.9663

# График ошибки от параметра k 

![img](./img/Screenshot_4.png)