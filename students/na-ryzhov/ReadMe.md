# Lab5
В качестве датасета для классификации был выбран датасет iris

Linear Kernel:<br>
  Native - Accuracy: 1.0000, Time: 2.9673 seconds <br>
  Sklearn - Accuracy: 1.0000, Time: 0.0010 seconds <br>
RBF Kernel:<br>
  Native - Accuracy: 0.4333, Time: 16.7492 seconds<br>
  Sklearn - Accuracy: 1.0000, Time: 0.0010 seconds<br>
Polynomial Kernel:<br>
  Native - Accuracy: 1.0000, Time: 2.5511 seconds<br>
  Sklearn - Accuracy: 1.0000, Time: 0.0000 seconds<br>

Все реализации, за исключением RBF показывают результаты идентичные с библиотечными по точности, но уступают по скорости.

![img](./img/Screenshot_1.png)