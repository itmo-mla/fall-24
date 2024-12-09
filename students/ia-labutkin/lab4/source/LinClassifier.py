# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:42:47 2024

@author: Ivan
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearClassifier():
    W=np.array([])
    W0=np.array([])
    activation_fn=np.sign
    def predict(self,X):#Предсказание класса объекта
        return self.activation_fn(np.matmul(self.W,X.T)+self.W0)
    def count_margins(self,X,y):#Вычисление отступов объектов
        return (np.matmul(self.W,X.T)+self.W0)*y
    def visualise_margins(self,X,y):#Визуализация отступов объектов
        plt.plot(np.sort(self.count_margins(X,y)))
        plt.xlabel('i')
        plt.ylabel('Margin')
        plt.show()
    def Q(self,X,y):#Вычисление функции потерь с регуляризацией
        return np.sum((1-self.count_margins(X,y))**2)+np.sum((self.W)**2)+(self.W0)**2
    def dM(self,X,y):#Вычисление производной от самого отступа
        return 2*(1-self.count_margins(X,y))
    def gradient(self,X,y,t):#Вычисление градиента для весов с регуляризацией
        return -np.matmul(self.dM(X,y),X*y.reshape(-1,1))+t*self.W
    def gradient_b(self,X,y,t):#Вычисление градиента для свободного члена с регуляризацией
        return np.sum(self.dM(X,y))+t*self.W0
    def init_weights_corellation(self,X,y):#Инициализация весов через корреляцию
        W=np.array([])
        for i in range(X.shape[1]):
            W=np.append(W,np.dot(X[:,i],y)/np.dot(X[:,i],X[:,i]))
        return W
    def count_accuracy(self,X,y):#Подсчёт точности классификации
        return np.sum(self.predict(X)==y)/len(y)
    def multistart(self,X,y,  t, inertion, lambda_rec, lr, margin_present, eps,n_starts=10):#Инициализация весов через мультистарт
        weights=[]
        accs=np.array([])
        indexes=np.random.choice(X.shape[0], X.shape[0]//2, replace = False)
        for i in range(n_starts):
            self.train_sgd(X[indexes],y[indexes],t=t, inertion=inertion, lambda_rec=lambda_rec, lr=lr, margin_present=margin_present, eps=eps)
            weights.append(self.W)
            accs=np.append(accs,self.count_accuracy(X,y))
        return weights[np.argmax(accs)]
        
    def train_sgd(self,X,y,t=0.01, inertion=0.8, lambda_rec=0.8, lr=0.0001, fast_descent=False, margin_present=True, init_weights='random', eps=0.001):#Обучение через стохастический градиентный спуск с инерцией
        if init_weights=='corellation':
            self.W=self.init_weights_corellation(X,y)
        elif init_weights=='multistart':
            self.W=self.multistart(X,y,t=t, inertion=inertion, lambda_rec=lambda_rec, lr=lr, margin_present=margin_present, eps=eps)
        else:
            self.W=np.random.uniform(-1/(2*X.shape[1]), 1/(2*X.shape[1]), size=X.shape[1])
        self.W0=np.random.uniform(-1/(2*X.shape[1]), 1/(2*X.shape[1]), size=1)
        if margin_present==True:#Предъявление объектов по модулю отступа
            M=self.count_margins(X,y)
            probs=(1/np.sum(1/np.abs(M))*(1/np.abs(M))).astype('float64')
            indexes=np.random.choice(len(probs), size = len(probs), replace = False, p = probs)
        else:#Случайное предъявление объектов
            indexes=np.random.choice(X.shape[0], X.shape[0], replace = False)
        idx=np.random.choice(X.shape[0], 10, replace=False)
        Q_start=self.Q(X[idx],y[idx])#Задание начального функционала качества
        v=np.zeros_like(self.W)#Инициализация инерций для весов
        v_b=np.zeros_like(self.W0)#Инициализация инерций для свободного члена
        Q_s=np.array([])
        Q_starts=np.array([])
        k=0
        for i in indexes:
            X_b=X[i].reshape(1,-1)
            y_b=np.array([y[i]])
            e=self.Q(X_b,y_b)#Loss на объекте
            v=inertion*v+(1-inertion)*self.gradient(X_b,y_b,t)#Добавление инерции
            v_b=inertion*v_b+(1-inertion)*self.gradient_b(X_b,y_b,t)#Добавление инерции
            if fast_descent==True:#Определение шага через скорейший градиентный спуск
                h=np.power(np.sum(X_b**2),-2)
            else:
                h=lr#Использование постоянного шага
            self.W-=h*v#Обновление весов
            self.W0-=h*v_b#Обновление свободных членов
            if np.all(abs(Q_start/(lambda_rec*e+(1-lambda_rec)*Q_start)-1)<eps):
                break
            Q_start=lambda_rec*e+(1-lambda_rec)*Q_start#Рекурентная оценка функционала качества
            Q_s=np.append(Q_s,e)
            Q_starts=np.append(Q_starts,Q_start)
            k+=1
            
        return Q_s,Q_starts 