# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

def split_data(ratings): # 90% training, 10% test
    n = len(ratings)
    n_train = round(9/10*n)
    ratings_random = ratings.copy()
    np.random.shuffle(ratings_random)
    data_train = ratings_random[:n_train]
    data_test = ratings_random[n_train:]
    return data_train,data_test


def init_matrix(nb_users,nb_movies,z,a,b):
    # a et b définis l'horizon de tirage a*(0,1) -b
    Nu = a*np.random.random((nb_users,z)) - b
    Ni = a*np.random.random((z,nb_movies)) - b
    Nu /= Nu.sum(axis = 0)[np.newaxis,:]
    return Nu, Ni


def calcul_error(Nu,Ni,data_test):  # calculer erreur 
    error = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        error += (np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)**2
    return error 


def calcul_mean_error(Nu,Ni,data_test): # calculer la moyenne de l'erreur
    error = calcul_error(Nu,Ni,data_test)
    return error/len(data_test)  

def calcul_error_mse(Nu,Ni,data_test):
    # 5 classe
    cpt = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        rating_predi = np.dot(Nu[idu-1,:],Ni[:,idi-1])
        cpt += (rating_predi - rating)
    return cpt/len(data_test)

def calcul_error_percent(Nu,Ni,data_test,error_threshold): # calculer la pourcentage d'erreur avec un seuil donné
    cpt = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        if abs(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating) > error_threshold :
            cpt += 1
    return cpt/len(data_test)

def calcul_error_5(Nu,Ni,data_test):
    # 5 classe
    cpt = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        rating_predi = round(np.dot(Nu[idu-1,:],Ni[:,idi-1]))
        if rating_predi == rating:
            cpt += 1
    return cpt/len(data_test)
    

def calcul_error_2(Nu,Ni,data_test):
    # sépare les notes en deux parties : +:4,5; -:1,2 et on ignore 3 
    cpt = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        rating_predi = round(np.dot(Nu[idu-1,:],Ni[:,idi-1]))
        if (rating == 4 or rating == 5) and (rating_predi == 1 or rating_predi == 2):
            cpt += 1 
        elif (rating == 1 or rating == 2) and (rating_predi == 4 or rating_predi == 5):
            cpt += 1 
    return cpt/len(data_test)


def SGD_simple(data_train,Nu,Ni,nb_iter,nb_norm,eps,eps_ui): 
    nb = len(data_train)
    for i in range(nb_iter):
        """ à changer si lecture de fichier change """
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)] 
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%nb_norm == 0 :
            Nu *= 1-eps_ui
            Ni *= 1-eps_ui
    return Nu,Ni

       
def SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold):  # descente de gradiant stochastique
    nb = len(data_train)
    train_error_histo = []
    test_error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%nb_norm == 0 :
            print(i)
            print(datetime.datetime.now())
            Nu *= 1-epsu
            Ni *= 1-epsi
            iter_histo.append(i)
            train_error_histo.append(calcul_error_percent(Nu,Ni,data_train,error_threshold))
            test_error_histo.append(calcul_error_percent(Nu,Ni,data_test,error_threshold))
        #Nu /= Nu.sum(axis = 0)[np.newaxis,:]
        #Ni /= Ni.sum(axis = 1)[:,np.newaxis]
    return iter_histo,train_error_histo,test_error_histo,Nu,Ni

def SGD_error_2(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi):  # descente de gradiant stochastique
    nb = len(data_train)
    train_error_histo = []
    test_error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%nb_norm == 0 :
            print(i)
            print(datetime.datetime.now())
            Nu *= 1-epsu
            Ni *= 1-epsi
            iter_histo.append(i)
            train_error_histo.append(calcul_error_2(Nu,Ni,data_train))
            test_error_histo.append(calcul_error_2(Nu,Ni,data_test))
    return iter_histo,train_error_histo,test_error_histo,Nu,Ni

def SGD_error_5(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi):  # descente de gradiant stochastique
    nb = len(data_train)
    train_error_histo = []
    test_error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%nb_norm == 0 :
            print(i)
            print(datetime.datetime.now())
            Nu *= 1-epsu
            Ni *= 1-epsi
            iter_histo.append(i)
            train_error_histo.append(calcul_error_5(Nu,Ni,data_train))
            test_error_histo.append(calcul_error_5(Nu,Ni,data_test))
    return iter_histo,train_error_histo,test_error_histo,Nu,Ni



def optim_eps(data_train,data_test,Nu_init,Ni_init,lower_bound,upper_bound,number,base_log = 2.0,nb_iter = 400000,nb_norm = 2000,error_threshold = 0.5):
    # lower_bound et upper_bound sous forme de 1e-3
    # number est le nombre de point à tester
    # la base de logarithme par défaut est à 2
    start = math.log(lower_bound)/math.log(base_log)
    stop = math.log(upper_bound)/math.log(base_log)
    eps = np.logspace(start, stop, num=number, base=base_log)
    eps_ui = np.logspace(start, stop, num=number, base=base_log)
    error_train = np.zeros((number,number))
    error_test = np.zeros((number,number))
    for i in range(number):
        for j in range(number):
            print(i,j)
            print(datetime.datetime.now()) 
            Nu,Ni = SGD_simple(data_train,Nu_init,Ni_init,nb_iter,nb_norm,eps[i],eps_ui[j])
            error_train[i][j] = calcul_error_percent(Nu,Ni,data_train,error_threshold)
            error_test[i][j] = calcul_error_percent(Nu,Ni,data_test,error_threshold)
    return eps,eps_ui,error_train,error_test

def Regularization_profil(users,movies,Nu,Ni,epsr):
    print("regularization")
    for e in users:
        l = len(e)
        if l == 0:
            continue
        else:
            indexi = np.random.randint(0,l)
            indexj = np.random.randint(0,l)
            Nu[e[indexi]-1] = Nu[e[indexi]-1] + epsr*Nu[e[indexj]-1]
            Nu[e[indexj]-1] = Nu[e[indexj]-1] + epsr*Nu[e[indexi]-1]
    for e in movies:
        l = len(e)
        if l == 0:
            continue
        else:
            indexi = np.random.randint(0,l)
            indexj = np.random.randint(0,l)
            Ni[:,e[indexi]-1] = Ni[:,e[indexi]-1] + epsr*Ni[:,e[indexj]-1]
            Ni[:,e[indexj]-1] = Ni[:,e[indexj]-1] + epsr*Ni[:,e[indexi]-1]
    return Nu,Ni


def SGD_regularization_profil(users,movies,data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,epsr):  # descente de gradiant stochastique
    nb = len(data_train)
    train_error_histo = []
    test_error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%2000 == 0:
            """ quand le échantillons sont de même catégorie, on les rapproche """
            Nu,Ni = Regularization_profil(users,movies,Nu,Ni,epsr)
        if i%nb_norm == 0 :
            print(i)
            print(datetime.datetime.now())
            Nu *= 1-epsu
            Ni *= 1-epsi            
            iter_histo.append(i)
            train_error_histo.append(calcul_error_5(Nu,Ni,data_train))
            test_error_histo.append(calcul_error_5(Nu,Ni,data_test))
    return iter_histo,train_error_histo,test_error_histo,Nu,Ni 



def Regularization_profil_1(users,movies,Nu,Ni,Au,Ai,epsr):
    # il y a une matrice Nu et un matrice Au, on se rapproche Nu et Au
    print("regularization")
    for i,e in enumerate(users):
        l = len(e)
        if l == 0:
            continue
        else:
            indexi = np.random.randint(0,l)
            Nu[e[indexi]-1] = Nu[e[indexi]-1] + epsr*Au[i]
            Au[i] = Au[i] + epsr*Nu[e[indexi]-1]
    for i,e in enumerate(movies):
        l = len(e)
        if l == 0:
            continue
        else:
            indexi = np.random.randint(0,l)
            Ni[:,e[indexi]-1] = Ni[:,e[indexi]-1] + epsr*Ai[:,i]
            Ai[:,i] = Ai[:,i] + epsr*Ni[:,e[indexi]-1]
    return Nu,Ni

def SGD_regularization_profil_1(users,movies,data_train,data_test,Nu,Ni,Au,Ai,nb_iter,nb_norm,eps,epsu,epsi,epsr):  # descente de gradiant stochastique
    nb = len(data_train)
    train_error_histo = []
    test_error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        if i%2000 == 0:
            """ quand le échantillons sont de même catégorie, on les rapproche """
            Nu,Ni = Regularization_profil_1(users,movies,Nu,Ni,Au,Ai,epsr)
        if i%nb_norm == 0 :
            print(i)
            print(datetime.datetime.now())
            Nu *= 1-epsu
            Ni *= 1-epsi            
            iter_histo.append(i)
            train_error_histo.append(calcul_error_5(Nu,Ni,data_train))
            test_error_histo.append(calcul_error_5(Nu,Ni,data_test))
    return iter_histo,train_error_histo,test_error_histo,Nu,Ni,Au,Ai
   
