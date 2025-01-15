# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 23:16:38 2024

@author: Ivan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:45:36 2024

@author: Ivan
"""
import numpy as np
import pandas as pd

def m_entropy(x):#Функция подсчёта энтропии
    if x==0:
        return 0
    else:
        return -x*np.log2(x)

def split_data(x,y,crit,value):#Функция разделения данных
    try: 
        x[:,crit].astype('float64')
        X_r=x[np.where(x[:,crit]>=value)]
        X_l=x[np.where(x[:,crit]<value)]
        y_r=y[np.where(x[:,crit]>=value)]
        y_l=y[np.where(x[:,crit]<value)]
    except ValueError:
        X_r=x[np.where(x[:,crit]==value)]
        X_l=x[np.where(x[:,crit]!=value)]
        y_r=y[np.where(x[:,crit]==value)]
        y_l=y[np.where(x[:,crit]!=value)]
    return X_r, X_l, y_r,y_l

def get_optimal_split(x,y, crit='e'):#Получение оптимального разбиения
    max_entropy=-1
    value=0
    feat=0
    for feature in range(x.shape[1]):
        nans_idx=np.where(pd.isnull(x[:,feature]))#Временное удаление пропусков из выборки
        nans_x=x[nans_idx]
        nans_y=y[nans_idx]
        x=np.delete(x, nans_idx, axis=0)
        y=np.delete(y, nans_idx, axis=0)
        values=np.unique(x[:,feature])[1:]#Выделение всех уникальных значений признака
        if crit=='d':#Критерий Донского
            for v in values:#Для каждого разделения смотрим информативность
                X_r, X_l, y_r,y_l=split_data(x,y,feature,v)#Разделение данных
                entropy=0
                for i in range(len(X_l)):
                    entropy+=np.sum(y_l[i]!=y_r)#Подсчёт разных значений для каждого элемента
                if entropy>max_entropy:#Если информативность наибольшая, запоминаем
                    feat=feature
                    value=v
                    max_entropy=entropy
        else:#Энтропийный критерий
            for v in values:#Для каждого разделения смотрим информативность
                X_r, _, y_r,_ = split_data(x,y,feature,v)#Разделение данных
                entropy=0
                for i in np.unique(y):#Подсчёт энтропии по формуле
                    l=x.shape[0]
                    p=X_r.shape[0]
                    Pc=y[np.where(y ==i)].shape[0]
                    pc=y_r[np.where(y_r ==i)].shape[0]
                    entropy+=m_entropy(Pc/l)-p/l*m_entropy(pc/p)-(l-p)/l*m_entropy((Pc-pc)/(l-p))
                if entropy>max_entropy:#Если информативность наибольшая, запоминаем
                    feat=feature
                    value=v
                    max_entropy=entropy
        x=np.concatenate((x,nans_x), axis=0)#Возвращение пропусков в выборки
        y=np.concatenate((y,nans_y), axis=0)
    nans_idx=np.where(pd.isnull(x[:,feat]))
    nans_x=x[nans_idx]
    nans_y=y[nans_idx]
    x=np.delete(x, nans_idx, axis=0)
    y=np.delete(y, nans_idx, axis=0)
    X_r, X_l, y_r,y_l=split_data(x,y,feat,value)
    q=len(X_l)/(len(X_r)+len(X_l))
    rans=np.array(np.random.choice([0,1], len(nans_x), replace=True, p=[q,1-q]))#Распределение пропусков по вероятностям их попадания в ту или иную ветвь
    X_l=np.concatenate((X_l,nans_x[np.where(rans==0)]),axis=0)
    y_l=np.concatenate((y_l,nans_y[np.where(rans==0)]),axis=0)
    X_r=np.concatenate((X_r,nans_x[np.where(rans==1)]), axis=0)
    y_r=np.concatenate((y_r,nans_y[np.where(rans==1)]),axis=0)
    return feat, value, (X_r, X_l, y_r,y_l)


def sum_dicts(d_1,d_2):#Суммирование словарей по ключам
    for i in d_1.keys():
        d_1[i]+=d_2[i]
    return d_1

def mse_crit(y, y_u):#Подсчёт MSE-критерия
    return np.min(np.sum(np.square(y.reshape(-1,1)-y_u),axis=1)/len(y_u))

def get_optimal_split_r(x,y,y_all):#Выбор оптимального разделения для регрессии
    max_entropy=-1
    value=0
    feat=0
    ent_start=mse_crit(y_all,y)
    for feature in range(x.shape[1]):
        nans_idx=np.where(pd.isnull(x[:,feature]))
        nans_x=x[nans_idx]
        nans_y=y[nans_idx]
        x=np.delete(x, nans_idx, axis=0)
        y=np.delete(y, nans_idx, axis=0)
        values=np.unique(x[:,feature])[1:]
        for v in values:#Реализация поиска оптимального разбиения по MSE-критерию
            _,_, y_r,y_l=split_data(x,y,feature,v)
            entropy=ent_start-mse_crit(y_all,y_r)*len(y_r)/len(y)-mse_crit(y_all,y_l)*len(y_l)/len(y)
            if entropy>max_entropy:
                feat=feature
                value=v
                max_entropy=entropy
        x=np.concatenate((x,nans_x), axis=0)
        y=np.concatenate((y,nans_y), axis=0)
    nans_idx=np.where(pd.isnull(x[:,feat]))
    nans_x=x[nans_idx]
    nans_y=y[nans_idx]
    x=np.delete(x, nans_idx, axis=0)
    y=np.delete(y, nans_idx, axis=0)
    X_r, X_l, y_r,y_l=split_data(x,y,feat,value)
    q=len(X_l)/(len(X_r)+len(X_l))
    rans=np.array(np.random.choice([0,1], len(nans_x), replace=True, p=[q,1-q]))
    X_l=np.concatenate((X_l,nans_x[np.where(rans==0)]),axis=0)
    y_l=np.concatenate((y_l,nans_y[np.where(rans==0)]),axis=0)
    X_r=np.concatenate((X_r,nans_x[np.where(rans==1)]), axis=0)
    y_r=np.concatenate((y_r,nans_y[np.where(rans==1)]),axis=0)
    return feat, value, (X_r, X_l, y_r,y_l)

def get_average_value(rules,level,key):#Получение среднего по листу значения
    vals=[]
    for spl in rules[level+1]:
        if spl[0]==key+'0' or spl[0]==key+'1':
            if spl[1]=='leaf':
                vals.append([spl[2],spl[3]])
            else:
                for v in get_average_value(rules,level+1,spl[0]):
                    vals.append(v)
    return vals

class DecisionTreeClassifier():#Дерево классификации
    rules=[]#Массив из правил
    splits=[]#Массив из разбиений
    uniques=[]#Массив из уникальных значений
    def fit(self, x,y, crit='e'):
        self.uniques=np.unique(y)
        splits=[[['0',x,y]]]
        rules=[]
        k=1
        while k==len(splits):#Пока дерево растёт в глубину, продолжаем
            p=[]
            r=[]
            for split in splits[k-1]:#Рассматриваем каждый уровень дерева
                u=np.unique(split[2])
                if len(u)==1:#Если всего один уникальный класс, то делаем лист
                    r.append([split[0], 'leaf', u[0], len(split[2])])
                else:#Иначе - ищем оптимальное разбиение, делаем узел и 2 разбиения на следующем уровне
                    feature, value, [X_r, X_l, y_r,y_l]= get_optimal_split(split[1],split[2], crit)
                    p.append([split[0]+'0',X_l,y_l])
                    p.append([split[0]+'1',X_r,y_r])
                    r.append([split[0], 'node', feature, value])
            rules.append(r)
            k+=1
            if len(p)>0:
                splits.append(p)
        self.splits=splits
        self.rules=rules
    def get_all_values(self, level, key):#Получение суммарного количества объектов каждого класса в листах уровнями ниже
        d={}
        for i in self.uniques:
            d[i]=0
        for spl in self.rules[level+1]:
            if spl[0]==key+'0' or spl[0]==key+'1':
                if spl[1]=='leaf':
                    d[spl[2]]+=spl[3]
                else:
                    d=sum_dicts(d,self.get_all_values(level+1,spl[0]))
        return d

        
    def predict(self,x):
        predict=[]
        for i in range(len(x)):
            xt=x[i,:]
            k_start='0'
            level=0
            flg=True
            while flg==True:
                for m in self.rules[level]:
                    if m[0]==k_start:
                        if m[1]=='leaf':#Если попали в лист, то выдаём класс, в который попали
                            predict.append(m[2])
                            flg=False
                        elif pd.isnull(xt[m[2]])==True:#Если пустое значение, то выдаём наиболее вероятное значение
                            d=self.get_all_values(level, k_start)
                            predict.append(max(zip(d.values(), d.keys()))[1])
                            flg=False
                        else:#Если попали в узел, то по предикату идём на следующий уровень
                            try:
                                if xt[m[2]]>=m[3]:
                                    k_start=k_start+'1'
                                else:
                                    k_start=k_start+'0'
                                level+=1 
                            except TypeError:
                                if xt[m[2]]==m[3]:
                                    k_start=k_start+'1'
                                else:
                                    k_start=k_start+'0'
                                level+=1
                        
        return predict
    
class DecisionTreeRegressor:#Дерево регрессии
    def __init__(self, depth):
        self.depth = depth
    rules=[]
    splits=[]
    def fit(self, x,y):
        splits=[[['0',x,y]]]
        rules=[]
        k=1
        while k<=self.depth and k==len(splits):#Пока дерево растёт вглубь и не достигнута максимальная глубина
            p=[]
            r=[]
            for split in splits[k-1]:
                feature, value, [X_r, X_l, y_r,y_l]= get_optimal_split_r(split[1],split[2], y)
                if len(y_r)==0 or len(y_l)==0:
                    r.append([split[0],'leaf',np.mean(split[2]),len(split[2])])#При попадании в лист считаем среднее значение таргетов элементов
                else:#При попадании в узел формируем новые ветки
                    p.append([split[0]+'0',X_l,y_l])
                    p.append([split[0]+'1',X_r,y_r])
                    r.append([split[0], 'node', feature, value])
            rules.append(r)
            k+=1
            if len(splits)<self.depth:
                splits.append(p)
        for i in range(len(splits[-1])):#По окончании цикла превращаем все разбиения на максимальной глубине в листья
            rules[-1][i]=[rules[-1][i][0],'leaf',np.mean(splits[-1][i][2]),len(splits[-1][i][2])]
        self.splits=splits
        self.rules=rules
        
    def predict(self,x):
            predict=[]
            for i in range(len(x)):
                xt=x[i,:]
                k_start='0'
                level=0
                flg=True
                while flg==True:
                    for m in self.rules[level]:
                        if m[0]==k_start:#При попаданиив лист присваеваем объекту среднее значение в листе
                            if m[1]=='leaf':
                                predict.append(m[2])
                                flg=False
                            elif pd.isnull(xt[m[2]])==True:#Если значение пустое, присваем средневзвешенное по листьям ниже узла
                                summa=np.sum(np.array([a[0]*a[1] for a in get_average_value(self.rules,level,k_start)]))
                                kol=np.sum(np.array([a[1] for a in get_average_value(self.rules,level,k_start)]))
                                predict.append(summa/kol)
                                flg=False    
                            else:#Иначе - идём дальше по дереву
                                try:
                                    if xt[m[2]]>=m[3]:
                                        k_start=k_start+'1'
                                    else:
                                        k_start=k_start+'0'
                                    level+=1 
                                except TypeError:
                                    if xt[m[2]]==m[3]:
                                        k_start=k_start+'1'
                                    else:
                                        k_start=k_start+'0'
                                    level+=1    
            return predict  
    
    