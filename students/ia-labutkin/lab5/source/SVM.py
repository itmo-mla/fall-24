# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:32:06 2024

@author: Ivan
"""

import numpy as np
from scipy.optimize import Bounds, minimize

class SVM():
    w=np.array([])
    w0=np.array([])
    def fit(self, x,y,c=2, d=3, gamma=-1, solution='linear'):
        self.solution=solution#Записываем вид решения
        lam=np.zeros_like(y)#Начальное приближение лямбды
        bnds=Bounds (np.zeros_like(lam), np.ones_like(lam) * c)#Ограничение на размер лямбды (регуляризация)
        sum_cons = {'type': 'eq',
                     'fun': lambda lam: lam@y
                    }#Ограничение на произведение лямбды и y
        if self.solution=='linear':#Оптимизируемая функция при линейном ядре
            value= lambda lam:-np.sum(lam)+np.sum(np.matmul((lam*y).reshape(-1,1)*x,((lam*y).reshape(-1,1)*x).T))
        elif self.solution=='rbf':#Оптимизация при ядре RBF
            if gamma==-1:
                self.gamma=1/(x.shape[1]*np.std(x))#Оптимальный гамма
            else:
                self.gamma=gamma
            value= lambda lam:-np.sum(lam)+np.sum(np.matmul((lam*y).reshape(-1,1),(lam*y).reshape(1,-1))*np.exp(-self.gamma*np.sum(np.square(x[:,None,:]-x[None,:,:]), axis=2)))
        else:#Оптимизация при полиномиальное ядро
            self.d=d
            value= lambda lam:-np.sum(lam)+np.sum(np.matmul((lam*y).reshape(-1,1),(lam*y).reshape(1,-1))*np.power(np.matmul(x,x.T),self.d))
        res = minimize(value, lam, method='SLSQP', #Решение оптимизационной задачи методом последовательного квадратичного программирования
                       constraints=sum_cons, bounds=bnds, options={'disp': True})
        not_null_lambda_num=np.where((res.x>1e-3) & (res.x<c))[0][0]#Нахождение ненулевой лямбды для получения w0
        if self.solution=='linear':
            w=np.dot(res.x*y,x)#Нахождение весов (только для линейного ядра)
            w0=np.dot(x[not_null_lambda_num],w)-y[not_null_lambda_num]#Нахождение свободного члена
            self.w=w
        elif self.solution=='poly':
            w0=np.sum(res.x*y*np.power(np.matmul(x,x[[not_null_lambda_num]].T),self.d))-y[not_null_lambda_num]#Нахождение свободного члена
        else:
            w0=np.sum((res.x*y).reshape(-1,1)*np.exp(-self.gamma*np.sum(np.square(x[:,None,:]-x[not_null_lambda_num]),axis=2)))-y[not_null_lambda_num]#Нахождение свободного члена
        #Сохранение переменных для предикта
        self.w0=w0
        self.x_t=x
        self.y_t=y
        self.lam=res.x
        
    def predict(self,x):
        if self.solution=='linear':
            return np.sign(np.dot(x,self.w)-self.w0)#Обычное умножение на весы при линейном ядре
        elif self.solution=='rbf':
            return np.sign(np.sum((self.lam*self.y_t).reshape(-1,1)*np.exp(-self.gamma*np.sum(np.square(self.x_t[:,None,:]-x[None,:,:]),axis=2)),axis=0))#Проход ядром по тестовой выборке
        else:
            return np.sign(np.sum((self.lam*self.y_t).reshape(-1,1)*np.power(np.matmul(self.x_t,x.T),self.d),axis=0)+self.w0)#Проход ядром по тестовой выборке
    def count_accuracy(self,x,y_real):
        return np.sum(y_real==self.predict(x))/len(y_real) #Подсчёт точности