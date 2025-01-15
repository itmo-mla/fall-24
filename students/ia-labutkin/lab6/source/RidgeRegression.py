# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:45:47 2025

@author: Ivan
"""
import numpy as np
import matplotlib.pyplot as plt

class RidgeRegressor:
    def __init__(self, fit_intercept):
        self.intercept=fit_intercept#Наличие свободного члена
    def fit(self,X,y, t=0):#Получение SVD-матриц, высчитывание вектора весов
        if self.intercept==True:
            X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)#Если есть свободный член, то добавляем слева столбец с единицами
        self.t=t#Параметр регуляризации
        fft=np.matmul(X,X.T)#Формирование матрицы F*F.T
        ftf=np.matmul(X.T,X)#Формирование матрицы F.T*F
        fft_eig_numbers,fft_eig_vectors=np.linalg.eig(fft)#Нахождение собственных векторов матрицы F*F.T
        self.U_vals,self.U=np.linalg.eig(ftf)#Нахождение собственных чисел и векторов матрицы F.T*F, т.е. составление матрицы U
        not_null_values=np.where(~np.isclose(fft_eig_numbers,0))#Нахождение индексов всех ненулевых собственных чисел матрицы F*F.T
        self.V=-fft_eig_vectors[:,not_null_values][:,0,:].astype('float64')#Нахождение всех собственных векторов для ненулевых собственных чисел матрицы F*F.T т.е. составление матрицы V
        #Приведение собственных векторов из матрицы V в порядок, соответствующий собственным числам из матрицы U
        idx=self.U_vals.argsort()
        self.V=self.V[:,idx[::-1]]
        self.D=np.diagflat(np.sqrt(self.U_vals)) #Формирование матрицы D
        self.W=self.U@np.linalg.inv(self.D**2+np.eye(self.D.shape[0])*self.t)@self.D@self.V.T@y#Нахождение вектора весов
    def fit_just_weights(self,y, t): #Перерасчитывание весов без изменения SVD-матрицы 
        self.t=t
        self.W=self.U@np.linalg.inv(self.D**2+np.eye(self.D.shape[0])*self.t)@self.D@self.V.T@y#Перерасчёт весов
    def count_Q(self, X_val, y_train, y_val, t): #Высчитывание функционала потерь на контрольной выборке
        if self.intercept==True:
            return np.linalg.norm(np.concatenate((np.ones((X_val.shape[0],1)),X_val),axis=1)@self.U@np.diagflat((np.sqrt(self.U_vals)/(self.U_vals+t)))@self.V.T@y_train-y_val)
        else:
            return np.linalg.norm(X_val@self.U@np.diagflat((np.sqrt(self.U_vals)/(self.U_vals+t)))@self.V.T@y_train-y_val)
    def choose_t(self, X_val,y_train,y_val,t_start,t_end,step):#Подбор оптимального параметра регуляризации
        Q_s=[]
        for i in range(t_start,t_end,step):
            Q_s.append(self.count_Q(X_val, y_train, y_val, i)) #Для каждого значения параметра считается функционал потерь
        plt.plot(range(t_start,t_end,step),Q_s)
        plt.title('Зависимость функционала ошибки от параметра регуляризации')
        plt.xlabel('Значение параметра регуляризации')
        plt.ylabel('Значение функции потерь')
        plt.grid()
        plt.show()
    def cond_number(self):#Расчёт числа обусловленности матрицы
        return np.max(self.U_vals)/np.min(self.U_vals)
    def predict(self, X):#Предсказание (умножение входящей матрицы на рассчитанные весы)
        if self.intercept==True:
            return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)@self.W
        else:
            return X@self.W