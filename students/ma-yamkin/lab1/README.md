Для выполнения первой лабораторной работы были выбраны датасеты с видами рыб и видами цветов ириса. Целевой признак, в котором и хранится разделение по кластерам, был удален. Количество кластеров было подобрано при помощи дендрограммы. Для датасета с видами рыб оно составило 9 кластеров, для датасета с видами ириса оно составило 4 кластера.

Бло реализовано три алгоритма кластеризации: иерархический, em и dbcscan. 

1. Точность иерархического алгоритма, реализованного вручную, совпадает с точностью библиотечного алгоритма. Также были построены дендрограммы по результатам ручной кластеризации. Данные дендрограммы полностью совпадают с библиотечными дендрограммамми. Однако время работы сильно отличается: составляет намного больше, чем время работы библиотечного алгоритма.
![image](https://github.com/user-attachments/assets/de8fb5e7-07fb-49a1-a174-761357608986)

Дендрограмма ручного алгоритма (датасет с рыбами)

![image](https://github.com/user-attachments/assets/26b101be-864b-43f5-a8cf-94f6f127a771)

Дендрограмма автоматического алгоритма (датасет с рыбами)

![image](https://github.com/user-attachments/assets/959a3695-c040-47bc-9bfa-bb555bee1982)
![image](https://github.com/user-attachments/assets/3165441d-ba6b-425a-a64d-a320b93a2016)
![image](https://github.com/user-attachments/assets/8796dc95-d954-43d3-9e9f-7a5d5bea782d)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/0e1d95ea-4c47-44d0-978f-51bce0f0fa46)
![image](https://github.com/user-attachments/assets/9912bb22-76f6-45da-8a30-5fc3762b12fa)

Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/589adf30-3cfe-4e9e-ab6b-888cc07ff29a)

Дендрограмма ручного алгоритма (датасет с ирисами)

![image](https://github.com/user-attachments/assets/adf070f8-c862-4a0d-a7ef-866b0934bd44)

Дендрограмма автоматического алгоритма (датасет с ирисами)

![image](https://github.com/user-attachments/assets/e5869995-e4da-475c-b277-c52066ebf9af)
![image](https://github.com/user-attachments/assets/50274f2e-930f-4f7e-90da-305a094ac9d2)
![image](https://github.com/user-attachments/assets/d0a289f3-b8bd-4cd6-8732-817c543a04bb)

Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/cfdceb53-ef19-45f8-99f7-9143fe579d68)
![image](https://github.com/user-attachments/assets/6da0d0f1-d108-4ef7-abbc-d4f5524e2a04)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

2. Точность em алгоритма несколько отличается от библиотечного алгоритма: это связано с тем, что алгоритм использует рандомные начальные приближения кластеров и является вероятностным, за счет каждый раз он срабатывает по разному. Однако время работы библиотечного алгоритма и алгоритма, написанного вручную, сравнимо.
   
![image](https://github.com/user-attachments/assets/df18a87a-beb6-4c73-9298-f12f604a7db7)
![image](https://github.com/user-attachments/assets/37783a6a-137e-4f10-a60f-a8c83bcbe1bf)
![image](https://github.com/user-attachments/assets/d61a82a5-5356-4cfd-976a-4762e50abbf7)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/04d43bfb-06be-4df6-9cfc-2147976b8823)
![image](https://github.com/user-attachments/assets/a6d085cc-1f75-45bb-8182-f745571d3e46)

Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/5194ef3a-75bb-4240-9001-a96a5c2bbeb0)
![image](https://github.com/user-attachments/assets/a0149d69-c770-451c-8b01-a402b073e89e)
![image](https://github.com/user-attachments/assets/2fa78946-1734-412a-b380-7d0392f9ae97)

Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/8afbe1bf-887e-495a-a45f-ac018c9a7522)
![image](https://github.com/user-attachments/assets/25f75ccd-dcc9-441b-9da2-75cde05d4189)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

3. Точность dbscan алгоритма практически полностью совпадает с точностью библиотечного алгоритма. Однако время работы библиотечного алгоритма также составляет намного меньше.

![image](https://github.com/user-attachments/assets/31051f73-8fe6-424a-97f0-ffc758d101df)
![image](https://github.com/user-attachments/assets/3f52eb8c-38d4-4413-86d7-cbb7ab5b9834)
![image](https://github.com/user-attachments/assets/b84f48df-4117-457d-b54a-b00480f31f68)

Сравнение метрик на датасете с рыбами

![image](https://github.com/user-attachments/assets/7cc54ba2-de3c-4780-8938-075b020acfb0)
![image](https://github.com/user-attachments/assets/87eb7b9b-25fa-46ab-8e04-b16854c99496)

Сравнение разделения на датасете с рыбами (график выше ручная кластеризация, график ниже библиотечная)

![image](https://github.com/user-attachments/assets/7e2ab1b2-a8af-4d5f-9da6-cc3a68246722)
![image](https://github.com/user-attachments/assets/be5a5dd3-4de5-449e-b523-b3c0a96e4dec)
![image](https://github.com/user-attachments/assets/7abfa0ab-8c4d-4c79-9c4e-89ca9de0a5fb)


Сравнение метрик на датасете с ирисами

![image](https://github.com/user-attachments/assets/cf6b1a22-6616-4a21-99d7-a1dbd15ccf5d)
![image](https://github.com/user-attachments/assets/e6d6bac0-2f84-4b03-8a82-2e4eecffea71)

Сравнение разделения на датасете с ирисами (график выше ручная кластеризация, график ниже библиотечная)

Итого, наиболее предпочтительным алгоритмом из всех написанных является иерархический, так как точность данного алгоритма сопоставима с библиотечным, а также метрики кластеризации являются оптимальными.
