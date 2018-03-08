# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:45:03 2018

@author: Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nmf

def test1():   
    # plotter erreur sur données de test et d'apprentissage
    ratings = np.genfromtxt('ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    eps = 5e-3 #10e-3,5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 10000   # nb_iter = 20 000  est suffisant selon le graphe 
    nb_norm = 100
    error_threshold = 1
    Nu,Ni = init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    plt.figure()
    plt.plot(iter_histo,test_error_histo)
    plt.show()
#   Nu_embedded = TSNE(n_components=2).fit_transform(Nu)
    plt.figure()
    plt.plot(Nu_embedded[:,0],Nu_embedded[:,1], 'b*')
    
def test2():
    # optimiser eps
    ratings = np.genfromtxt('ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = split_data(ratings)
    eps,eps_ui,error_train,error_test = optim_eps(data_train,data_test,1e-3,1e-2,3)
    error_min = error_test.min()
    index = np.argmin(error_test)
    print(error_min)
    print(index)