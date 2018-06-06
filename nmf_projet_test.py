# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nmf
import pickle

def test1():  
    """ trouver minimum de nb itérations nécessaire """
    # plotter erreur sur données de test et d'apprentissage
    ratings = np.genfromtxt('data/data1/ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    eps = 5e-3 #10e-3,5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 10000   # nb_iter = 20 000  est suffisant selon le graphe 
    nb_norm = 100
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    plt.figure()
    plt.plot(iter_histo,test_error_histo)
    plt.show()
#    Nu_embedded = TSNE(n_components=2).fit_transform(Nu)
#    plt.figure()
#    plt.plot(Nu_embedded[:,0],Nu_embedded[:,1], 'b*')
    
def test2():
    """ trouver la meilleure couple de eps et eps_ui """
    ratings = np.genfromtxt('data/data1/ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    lb = 1e-5
    up = 1e-1
    nb_eps = 2
    eps,eps_ui,error_train,error_test = nmf.optim_eps(data_train,data_test,Nu,Ni,lb,up,nb_eps)
    np.save("result/data1/eps.npy",eps)
    np.save("result/data1/error_train.npy",error_train)
    np.save("result/data1/error_test",error_test)
    error_min = error_test.min()
    print("erreur minimale = ", error_min)
    f = plt.figure()
    plt.imshow(error_test,extent=[lb,up,lb,up])
    plt.colorbar()
    plt.title("error_test selon eps")
    plt.xlabel("epsilon de SGD")
    plt.ylabel("epsilon utilisateur idem")
    f.savefig("result/data1/erreur_eps.pdf")


""" nouvelle données avec plus d'information sur les utilisateurs"""
""" UserIDs range between 1 and 6040; MovieIDs range between 1 and 3952 """
def test3():
    """ trouver minimum de nb itérations nécessaire """
    """ 300 000 iter """
    ratings = np.load('data/data2/ratings.npy') 
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 128
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    nb_iter = 1000000
    nb_norm = 2000
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    np.save("result/data2/dim_128_full/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_128_full/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_128_full/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_128_full/Nu.npy",Nu)
    np.save("result/data2/dim_128_full/Ni.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128_full/error_percent.pdf")

def test4():
    """ trouver meilleure eps """ 
    """ error_test[39][8], SGD eps[39] = 1.5e-2 Reg eps_ui[8] = 4.5e-5 """
    ratings = np.load('data/data2/ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 3   
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    lb = 1e-5
    up = 1e-1
    nb_eps = 50
    nb_iter = 300000
    eps,eps_ui,error_train,error_test = nmf.optim_eps(data_train,data_test,Nu,Ni,lb,up,nb_eps,nb_iter = nb_iter)
    np.save("result/data2/eps.npy",eps)
    np.save("result/data2/error_train.npy",error_train)
    np.save("result/data2/error_test.npy",error_test)
    error_min = error_test.min()
    index_min = np.argmin(error_test)
    index_0 = index_min//error_test.shape[0]
    index_1 = index_min % error_test.shape[1]
    eps_best = eps[39]
    eps_ui_best = eps[8]
    print("erreur minimale = ", error_min,"index:",index_0,index_1)
    print("best eps :",eps_best," best eps_ui :",eps_ui_best)
    plt.figure()
    plt.imshow(error_test,extent=[lb,up,lb,up])
    plt.colorbar()
    plt.title("error_test selon eps")
    plt.xlabel("epsilon de SGD")
    plt.ylabel("epsilon utilisateur idem")
    plt.savefig("result/data2/error_eps.pdf")


def test5():
    """ sub_ratings with 600 users and 300 films , dim = 90"""
    ratings = np.load('data/data2/sub_ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 600
    nb_movies = 300
    z = 90   
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    nb_iter = 100000
    nb_norm = 200
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    np.save("result/data2/dim_90/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_90/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_90/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_90/Nu.npy",Nu)
    np.save("result/data2/dim_90/Ni.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_90/error_percent.pdf")


def test6():
    """ recherche de eps pour sub_ratings dim = 90 """    
    ratings = np.load('data/data2/sub_ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 600
    nb_movies = 300
    z = 90   
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,5,0)
    lb = 1e-5
    up = 1e-1
    nb_eps = 3
    nb_iter = 40000
    eps,eps_ui,error_train,error_test = nmf.optim_eps(data_train,data_test,Nu,Ni,lb,up,nb_eps,nb_iter = nb_iter)
    np.save("result/data2/dim_90/eps.npy",eps)
    np.save("result/data2/dim_90/error_train.npy",error_train)
    np.save("result/data2/dim_90/error_test.npy",error_test)
    error_min = error_test.min()
    index_min = np.argmin(error_test)
    index_0 = index_min//error_test.shape[0]
    index_1 = index_min % error_test.shape[1]
    #eps_best = eps[39]
    #eps_ui_best = eps[8]
    print("erreur minimale = ", error_min,"index:",index_0,index_1)
    #print("best eps :",eps_best," best eps_ui :",eps_ui_best)
    plt.figure()
    plt.imshow(error_test,extent=[lb,up,lb,up])
    plt.colorbar()
    plt.title("error_test selon eps")
    plt.xlabel("epsilon de SGD")
    plt.ylabel("epsilon utilisateur idem")
    plt.savefig("result/data2/dim_90/error_eps.pdf")
    
def test7():
    """ sub_ratings with 600 users and 300 films, dim = 128 """
    ratings = np.load('data/data2/sub_ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 600
    nb_movies = 300
    z = 128  
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    nb_iter = 100000
    nb_norm = 200
    #error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_2(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi)
    np.save("result/data2/dim_128/iter_histo2.npy",iter_histo)
    np.save("result/data2/dim_128/test_error_histo2.npy",test_error_histo)
    np.save("result/data2/dim_128/train_error_histo2.npy",train_error_histo)
    np.save("result/data2/dim_128/Nu2.npy",Nu)
    np.save("result/data2/dim_128/Ni2.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128/error_percent2.pdf")

def test8():
    """ sub_ratings with 600 users and 300 films, dim = 64 """
    ratings = np.load('data/data2/sub_ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 600
    nb_movies = 300
    z = 64  
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    nb_iter = 100000
    nb_norm = 200
    error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_included(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,error_threshold)
    np.save("result/data2/dim_64/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_64/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_64/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_64/Nu.npy",Nu)
    np.save("result/data2/dim_64/Ni.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_64/error_percent.pdf")

def test9():
    """ dim = 128 est meilleure """
    erreur90 = np.load("result/data2/dim_90/test_error_histo.npy")
    erreur64 = np.load("result/data2/dim_64/test_error_histo.npy")
    erreur_diff = erreur90 - erreur64
    plt.plot(erreur_diff)
    erreur128 = np.load("result/data2/dim_128/test_error_histo.npy")
    erreur_diff2 = erreur128 - erreur90
    plt.plot(erreur_diff2)
    erreur_diff3 = erreur128 - erreur64
    plt.plot(erreur_diff3)

def test10():
    """ Régularization selon profil """
    ratings = np.load('data/data2/sub_ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 600
    nb_movies = 300
    z = 128  
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    epsr = 1e-4
    nb_iter = 100000
    nb_norm = 2000
    #error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    #Au,Ai = nmf.init_matrix(nb_users_att,nb_movies_att,z,1,0)
    with open ("data/data2/liste_sub_movies_genre.pkl", 'rb') as fp:
        list_movies = pickle.load(fp)
    with open ("data/data2/liste_sub_users.pkl", 'rb') as fp:
        list_users = pickle.load(fp)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_regularization_profil(list_users,list_movies,data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,epsr)    
    """ li_att_users : 0-1:sexe; 2-8:age; 9-29:occupation """
    #li_att_users = ['F','M',1,18,25,35,45,50,56,0,1,2,3,4,5,6,7,8,9,10,11,\
    #               12,13,14,15,16,17,18,19,20]
    """ li_att_users : genres film """
#    li_att_movies = ["Action","Adventure","Animation","Children's","Comedy",\
#                     "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",\
#                     "Musical","Mystery","Romance","Sci-Fi","Thriller","War","West"]
    np.save("result/data2/dim_128_e2_r/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_128_e2_r/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_128_e2_r/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_128_e2_r/Nu.npy",Nu)
    np.save("result/data2/dim_128_e2_r/Ni.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128_e2_r/error_percent.pdf")

def test11():
    """ Régularization selon profil pour le 5-classe """
    ratings = np.load('data/data2/ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 128  
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    epsr = 1e-4
    nb_iter = 1000000
    nb_norm = 2000
    #error_threshold = 1
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    #Au,Ai = nmf.init_matrix(nb_users_att,nb_movies_att,z,1,0)
    with open ("data/data2/liste_movies_genre.pkl", 'rb') as fp:
        list_movies = pickle.load(fp)
    with open ("data/data2/liste_users.pkl", 'rb') as fp:
        list_users = pickle.load(fp)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_regularization_profil(list_users,list_movies,data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi,epsr)    
    """ li_att_users : 0-1:sexe; 2-8:age; 9-29:occupation """
    #li_att_users = ['F','M',1,18,25,35,45,50,56,0,1,2,3,4,5,6,7,8,9,10,11,\
    #               12,13,14,15,16,17,18,19,20]
    """ li_att_users : genres film """
#    li_att_movies = ["Action","Adventure","Animation","Children's","Comedy",\
#                     "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",\
#                     "Musical","Mystery","Romance","Sci-Fi","Thriller","War","West"]
    np.save("result/data2/dim_128_e5_r/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_128_e5_r/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_128_e5_r/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_128_e5_r/Nu.npy",Nu)
    np.save("result/data2/dim_128_e5_r/Ni.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128_e5_r/error_percent.pdf")

def test12():
    """ Régularization selon profil pour le 5-classe """
    """ avec une matrice de l'utilisateur et une matrice de films """
    ratings = np.load('data/data2/ratings.npy')
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 128  
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    epsr = 1e-4
    nb_iter = 1000000
    nb_norm = 2000
    nb_users_att =  2 + 21 + 7
    nb_movies_att = 18
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    Au,Ai = nmf.init_matrix(nb_users_att,nb_movies_att,z,1,0)
    with open ("data/data2/liste_movies_genre.pkl", 'rb') as fp:
        list_movies = pickle.load(fp)
    with open ("data/data2/liste_users.pkl", 'rb') as fp:
        list_users = pickle.load(fp)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni,Au,Ai = nmf.SGD_regularization_profil_1(list_users,list_movies,data_train,data_test,Nu,Ni,Au,Ai,nb_iter,nb_norm,eps,epsu,epsi,epsr)    
    """ li_att_users : 0-1:sexe; 2-8:age; 9-29:occupation """
    #li_att_users = ['F','M',1,18,25,35,45,50,56,0,1,2,3,4,5,6,7,8,9,10,11,\
    #               12,13,14,15,16,17,18,19,20]
    """ li_att_users : genres film """
#    li_att_movies = ["Action","Adventure","Animation","Children's","Comedy",\
#                     "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",\
#                     "Musical","Mystery","Romance","Sci-Fi","Thriller","War","West"]
    np.save("result/data2/dim_128_e5_r1/iter_histo.npy",iter_histo)
    np.save("result/data2/dim_128_e5_r1/test_error_histo.npy",test_error_histo)
    np.save("result/data2/dim_128_e5_r1/train_error_histo.npy",train_error_histo)
    np.save("result/data2/dim_128_e5_r1/Nu.npy",Nu)
    np.save("result/data2/dim_128_e5_r1/Ni.npy",Ni)
    np.save("result/data2/dim_128_e5_r1/Au.npy",Au)
    np.save("result/data2/dim_128_e5_r1/Ai.npy",Ai)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128_e5_r1/error_percent.pdf") 

def test13():
    """ trouver minimum de nb itérations nécessaire """
    """ 300 000 iter """
    ratings = np.load('data/data2/ratings.npy') 
    data_train,data_test = nmf.split_data(ratings)
    nb_users = 6040
    nb_movies = 3883
    z = 128
    eps = 1.5e-2
    epsu = 4.5e-5
    epsi = 4.5e-5
    nb_iter = 1000000
    nb_norm = 2000
    Nu,Ni = nmf.init_matrix(nb_users,nb_movies,z,1,0)
    iter_histo,train_error_histo,test_error_histo,Nu,Ni = nmf.SGD_error_5(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi)
    np.save("result/data2/dim_128_full/iter_histo5.npy",iter_histo)
    np.save("result/data2/dim_128_full/test_error_histo5.npy",test_error_histo)
    np.save("result/data2/dim_128_full/train_error_histo5.npy",train_error_histo)
    np.save("result/data2/dim_128_full/Nu5.npy",Nu)
    np.save("result/data2/dim_128_full/Ni5.npy",Ni)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(iter_histo,train_error_histo)
    ax1.set_title('train')
    ax1.set_ylabel('poucentage mal prédit')
    ax2.plot(iter_histo,test_error_histo)
    ax2.set_title('test')
    ax2.set_xlabel('nb iteration')
    f.savefig("result/data2/dim_128_full/error_percent5.pdf")
test13()