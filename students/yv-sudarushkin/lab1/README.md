# Отчет по заданию

## Выбор датасетов
1. Для выполнения задания были выбраны следующие датасеты:
   - [**Вино**](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering): Датасет содержит химические параметры вин и их принадлежность к разным классам. Для кластеризации метки классов были удалены.
   - [**Цветы ирисы**](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset): Датасет с размерами лепестков и чашелистиков ирисов. Метки классов также были удалены из файла.
   
2. Визуализация данных для каждого из датасетов произведена с использованием алгоритма t-SNE. Результаты визуализации представлены на графиках для каждого алгоритма кластеризации:
   - ![image](https://github.com/user-attachments/assets/bb02e246-d6c7-4746-ae94-958eef541c53)
   - ![image](https://github.com/user-attachments/assets/d04ee963-dcbe-4313-a4d3-fcb52ceabd9c)

3. Тип кластеров:
   - Для обоих датасетов предполагается наличие компактных, сферических кластеров.
   
4. Гипотеза о количестве кластеров:
   - Для **Iris Dataset**: предположительно, 2-3 кластера.
   - Для **Wine Dataset**: предположительно, 3 кластера.

---

## Реализация алгоритмов кластеризации
1. Реализованы следующие алгоритмы:
   - Иерархическая кластеризация (Ward linkage).
   - EM-алгоритм (Gaussian Mixture Model).
   - DBSCAN (параметры: `epsilon=1.5, min_samples=5` для Iris; `epsilon=2.3, min_samples=12` для Wine).

2. Для каждого алгоритма произведены:
   - Построение кластеров.
   - Визуализация результатов кластеризации (графики t-SNE).
   - Оценка метрик кластеризации:
     - Среднее внутрикластерное расстояние.
     - Среднее межкластерное расстояние.
     - Время выполнения алгоритма.

---

## Результаты кластеризации

### Иерархический алгоритм
**Iris Dataset**:
- **Custom метрики**: {'mean_intra_d': 1.2094, 'mean_inter_d': 3.0526}
- **Custom Time**: 0.0399 сек.
- **SkLearn метрики**: {'mean_intra_d': 1.2094, 'mean_inter_d': 3.0526}
- **SkLearn Time**: 0.0031 сек.

![image](https://github.com/user-attachments/assets/32d2b049-b7f0-4898-9d10-2ff801b1cd40)
![image](https://github.com/user-attachments/assets/d8c54e3b-17b7-4729-a094-9a06bbed51ee)

**Wine Dataset**:
- **Custom метрики**: {'mean_intra_d': 3.6174, 'mean_inter_d': 5.5245}
- **Custom Time**: 0.0510 сек.
- **SkLearn метрики**: {'mean_intra_d': 3.6174, 'mean_inter_d': 5.5245}
- **SkLearn Time**: 0.0020 сек.

![image](https://github.com/user-attachments/assets/06edbaad-f433-43d0-9a01-8bf90e051346)
![image](https://github.com/user-attachments/assets/1ee14b1e-8cad-49c8-bb55-1519d3e3ed65)

---

### EM-алгоритм
**Iris Dataset**:
- **Custom метрики**: {'mean_intra_d': 1.8155, 'mean_inter_d': 3.0212}
- **Custom Time**: 0.0157 сек.
- **SkLearn метрики**: {'mean_intra_d': 1.4135, 'mean_inter_d': 3.6570}
- **SkLearn Time**: 0.3468 сек.

![image](https://github.com/user-attachments/assets/a55e6c2e-d1ef-4107-b10d-ed9f0f65577c)

**Wine Dataset**:
- **Custom метрики**: {'mean_intra_d': 3.5200, 'mean_inter_d': 5.5619}
- **Custom Time**: 0.0200 сек.
- **SkLearn метрики**: {'mean_intra_d': 3.5718, 'mean_inter_d': 5.5654}
- **SkLearn Time**: 0.4394 сек.

![image](https://github.com/user-attachments/assets/c55b88fb-c7eb-463b-bb40-8db9c6baf864)

---

### DBSCAN
**Iris Dataset**:
- **Custom метрики**: {'mean_intra_d': 1.4135, 'mean_inter_d': 3.6570}
- **Custom Time**: 0.0040 сек.
- **SkLearn метрики**: {'mean_intra_d': 1.4135, 'mean_inter_d': 3.6570}
- **SkLearn Time**: 0.0050 сек.

![image](https://github.com/user-attachments/assets/e7825a88-d347-4824-879f-69e15cc26a02)

**Wine Dataset**:
- **Custom метрики**: {'mean_intra_d': 1.4135, 'mean_inter_d': 3.6570}
- **Custom Time**: 0.0040 сек.
- **SkLearn метрики**: {'mean_intra_d': 1.4135, 'mean_inter_d': 3.6570}
- **SkLearn Time**: 0.0050 сек.

![image](https://github.com/user-attachments/assets/22f182ed-efa0-488d-adb0-4d07f22057fe)

---

## Сравнение метрик
Результаты для всех алгоритмов показывают, что кастомные реализации в целом демонстрируют схожие метрики с эталонными. Однако:
- Время выполнения кастомных реализаций для иерархического алгоритма выше из-за особенностей оптимизации Scikit-learn.
- EM-алгоритм показал незначительную разницу в значениях внутрикластерных расстояний между кастомной и эталонной реализацией для Iris Dataset.
- DBSCAN практически идентичен по результатам как для кастомной, так и для эталонной реализации.

---

## Выводы
1. Все три алгоритма кластеризации успешно реализованы и протестированы на двух датасетах.
2. Иерархическая кластеризация наиболее точно воспроизводит результаты эталонных реализаций.
3. EM-алгоритм имеет небольшие отличия, которые могут быть вызваны разными параметрами начальной инициализации.
4. DBSCAN демонстрирует стабильные результаты для кастомной и эталонной версий, но требует подбора гиперпараметров.
5. Кастомные реализации справляются с задачами, порой даже быстрее эталонных.
