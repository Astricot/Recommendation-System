#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:48:29 2018



"""
import clustering as cls
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import string


from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_notebook, save


def write_tsne():
#    f1 = 'data/data2/Nu.npy'
    f1 = 'data/data2/dim_90/Nu.npy'
#    f2 = 'data/data2/Ni.npy'   
    f2 = 'data/data2/dim_90/Ni.npy'
#    f3 = 'data/data2/Nu_embed.npy'
#    f4 = 'data/data2/Ni_embed.npy' 
    f3 = 'data/data2/dim_90/Nu_embed.npy'
    f4 = 'data/data2/dim_90/Ni_embed.npy'
    cls.tSNE_Nu(f1,f3)
    cls.tSNE_Ni(f2,f4)

def movie_genre(list_genre, Ni_embed, movies):
    movies_index = []
    for i in range(len(list_genre)):
        movies_index.append(movies[movies['genres'].str.contains(list_genre[i])].index.values)
        Ni_emb_mov =  Ni_embed.take(movies_index[i],axis=0)
#        nom = "result/data2/movies_genre/" + list_genre[i] + ".pdf"
        nom = "result/data2/sub_movies_genre/" + list_genre[i] + ".pdf"
        plt.figure()
        plt.scatter(Ni_embed[:,0], Ni_embed[:,1],c=class_Ni, marker='s', s=20)
        plt.scatter(Ni_emb_mov[:,0], Ni_emb_mov[:,1],c='red', marker='o', s=10)
        plt.savefig(nom)
        plt.show()
        plt.close()
        

#mode : "users_age", "users_gender", "users_occupation", "movies_genre", "sub_users_age", "sub_users_gender", "sub_users_occupation", "sub_movies_genre"
def cluster_analiysis(liste, N_embed, dataframe, class_N, nb_cluster, count_total, mode):
    result = []
    str_write = ""
    tmp_class = np.zeros((len(liste), nb_cluster))
    list_result = []    
    
    for i in range(len(liste)):
        #mode : for filtering dataframe by a specific atribute
        if "users_age" in mode:
            result.append(dataframe[(dataframe['Age'] >= liste[i][0]) & (dataframe['Age'] < liste[i][1]) ].index.values.tolist())
        if "users_gender" in mode:
            result.append(dataframe[dataframe['Gender'] == liste[i] ].index.values.tolist())
        if "users_occupation" in mode:
            result.append(dataframe[dataframe['Occupation'] == liste[i] ].index.values.tolist())
        if "movies_genre" in mode:
            result.append(dataframe[dataframe['genres'].str.contains(liste[i])].index.values.tolist())
        
        category_variable = 'category_' + str(i)
        category_variable = result[i] 
        new_result = [x+1 for x in result[i]]
        list_result.append(new_result)
        
        for k in range(len(category_variable)):
            tmp_class[i][class_N[category_variable[k]]] += 1
            
        N_emb_filtered =  N_embed.take(result[i],axis=0)
        
        #save cluster visuals for each category
        path = "result/data2/"+ mode + "/" + str(liste[i]) + ".pdf"
        plt.figure()
        plt.scatter(N_embed[:,0], N_embed[:,1],c=class_N, marker='s', s=20)
        plt.scatter(N_emb_filtered[:,0], N_emb_filtered[:,1],c='red', marker='o', s=10)        
        plt.savefig(path)
        plt.show()
        plt.close() 
        
    for c in range(nb_cluster):
#        print("Cluster ", c) 
        str_write += "Cluster " + str(c) + '\n'
        for a in range(len(liste)):
#            print("Age group", a, ":", (tmp_class.transpose()[c][a] / count_total[c]) * 100 , "%")
#            print(tmp_class.transpose()[c][a], count_total[c])
            str_write += "Category " +  str(liste[a]) + ": "+ str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + "%" + '\n'
#            str_write += str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + '\n'

#        print(str_write)
        path_for_result = "result/data2/" + mode + "/result.txt"
        file = open(path_for_result, "w")
        file.write(str_write) 
        file.close()
    path_list = "result/data2/" + mode + "/liste_"+ mode +".pkl"
#    file = open(path_list, "w")
#    file.writelines(["%s\n" % item  for item in list_result])
    with open(path_list, 'wb') as fp:
        pickle.dump(list_result, fp)  
    return list_result
    
   
def cluster_analiysis_2(liste1, liste2, N_embed, dataframe, class_N, nb_cluster, count_total, mode):
    result = []
    result2 = []
    str_write = ""
    tmp_class = np.zeros((len(liste1), nb_cluster))
    list_result = []    
    
    tmp_class2 = np.zeros((len(liste2), nb_cluster))
    list_result2 = []
    
    for i in range(len(liste2)):
        
        result2.append(dataframe[(dataframe['Age'] >= liste2[i][0]) & (dataframe['Age'] < liste2[i][1]) ].index.values.tolist())

        category_variable2 = 'category2_' + str(i)
        category_variable2 = result2[i] 
        new_result2 = [x+1 for x in result2[i]]
        list_result2.append(new_result2)
            
        for k in range(len(category_variable2)):
            tmp_class2[i][class_N[category_variable2[k]]] += 1
            
        N_emb_filtered2 =  N_embed.take(result2[i],axis=0)
        
        for t in range(len(liste1)):
            #mode : for filtering dataframe by a specific atribute
            result.append(dataframe[dataframe['Gender'] == liste1[t] ].index.values.tolist())
            
            category_variable = 'category_' + str(t)
            category_variable = result[t] 
            new_result = [x+1 for x in result[t]]
            list_result.append(new_result)
            
            for k in range(len(category_variable)):
                tmp_class[t][class_N[category_variable[k]]] += 1
                
            N_emb_filtered =  N_embed.take(result[t],axis=0)
            
        
            xx = ''.join(random.choice(string.ascii_letters) for m in range(2))
        
        
        
            #save cluster visuals for each category
            path = "result/data2/2/" + str(liste2[i]) + xx + ".pdf"
            plt.figure()
            plt.scatter(N_embed[:,0], N_embed[:,1],c=class_N, marker='s', s=20)
            plt.scatter(N_emb_filtered[:,0], N_emb_filtered[:,1],c='red', marker='o', s=10)        
            plt.scatter(N_emb_filtered2[:,0], N_emb_filtered2[:,1],c='white', marker='o', s=10)        
            plt.savefig(path)
            plt.show()
            plt.close() 
        
#    for c in range(nb_cluster):
##        print("Cluster ", c) 
#        str_write += "Cluster " + str(c) + '\n'
#        for a in range(len(liste)):
##            print("Age group", a, ":", (tmp_class.transpose()[c][a] / count_total[c]) * 100 , "%")
##            print(tmp_class.transpose()[c][a], count_total[c])
#            str_write += "Category " +  str(liste[a]) + ": "+ str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + "%" + '\n'
##            str_write += str((tmp_class.transpose()[c][a] / count_total[c]) * 100) + '\n'
#
##        print(str_write)
#        path_for_result = "result/data2/" + mode + "/result.txt"
#        file = open(path_for_result, "w")
#        file.write(str_write) 
#        file.close()


#    path_list = "result/data2/" + mode + "/liste_"+ mode +".pkl"
#    with open(path_list, 'wb') as fp:
#        pickle.dump(list_result, fp) 
#    return list_result    


def read_liste(mode):
    list_result = []
    path_list = "result/data2/" + mode + "/liste_"+ mode +".pkl"
    with open (path_list, 'rb') as fp:
        list_result = pickle.load(fp)
    return list_result
    
#ll = read_liste("sub_users_age")
 
        
def bokeh(users, Nu_embed, c_u):
    df_combine = users
    df_combine['x-tsne'] = Nu_embed[:,0]
    df_combine['y-tsne'] = Nu_embed[:,1]

    palette ={1:'red',2:'green',3:'blue',
              4:'yellow',5:"orange", 6:"pink",
              7:"black", 8:"purple", 9:"brown",0:"white"}
    colors =[]
    for i in c_u:
            colors.append(palette[i])

    print(colors)

    source = ColumnDataSource(dict(
        x=df_combine['x-tsne'],
        y=df_combine['y-tsne'],
        age= df_combine['Age'],
        gender= df_combine['Gender'],
        occupation = df_combine['Occupation'],
        color = colors
        ))

    hover_tsne = HoverTool( tooltips=[
                                     ("Age", "@age"),
                                     ("Gender", "@gender"),
                                     ("Occupation", "@occupation")
                                     ])
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']

    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='users')

    plot_tsne.circle('x', 'y', size=10, fill_color='color',
                     alpha=0.5, line_width=0, source=source, name="bokeh")

    # hover tools
    hover = plot_tsne.select(dict(type=HoverTool))
    hover.tooltips = {"content": "Age: @age, Gender: @gender, Occupation: @occupation"}

    show(plot_tsne)
    
def find_avg_age(nb_cluster, Nu_embed):
    for i in range(nb_cluster):
        avg = 0
        cl = np.where(class_Nu == i)
    #    nu_cl_zero = Nu_embed.take(cl_zero,axis=0)
        users_nb_cluster = users.iloc[cl]
        avg = users_nb_cluster['Age'].mean()
        median = users_nb_cluster['Age'].median()
        print("For cluster ",i, " age moyenne: ", avg )
        print("For cluster ",i, " age median: ", median )
   
def fing_gender_major(nb_cluster, Nu_embed):
    for i in range(nb_cluster):
        f = 0
        m = 0
        cl = np.where(class_Nu == i)
    #    nu_cl_zero = Nu_embed.take(cl_zero,axis=0)
        users_nb_cluster = users.iloc[cl]
        for j in users_nb_cluster["Gender"]:
            
            if j == "F":
                f += 1
            elif j == "M":
                m += 1
        print("For cluster ",i, " f: ", f, "m: ", m, "diff", f-m)
        
#def get_ratings_pd():
#    dtype = [('UserID','int'), ('MovieID','int'), ('Rating','int'), ('Timestamp','int')]
#    values = np.load("data/data2/ratings.npy")
#    values.astype(dtype)
#    index = [i for i in range(1, len(values)+1)]
#    ratings = pd.DataFrame(values, index=index)
#    ratings.rename(columns = {'f0':'UserID','f1':'MovieID','f2':'Rating','f3':'Timestamp'}, inplace=True)
#    return ratings

#f1 = 'data/data2/Nu_embed.npy'
#f1 = 'data/data2/dim_90/Nu_embed.npy'
#Nu_embed = np.load(f1)
#nb_cluster = 20
#class_Nu = cls.clustering(Nu_embed,nb_cluster)
#plt.figure()
#plt.scatter(Nu_embed[:,0], Nu_embed[:,1],c=class_Nu, marker='s', s=20)
#plt.show()

#f2 = 'data/data2/Ni_embed.npy'
#f2 = 'data/data2/dim_90/Ni_embed.npy'
#Ni_embed = np.load(f2)
#class_Ni = cls.clustering(Ni_embed,nb_cluster)
#plt.figure()
#plt.scatter(Ni_embed[:,0], Ni_embed[:,1],c=class_Ni, marker='s', s=20)
#plt.show()


#count_total_u = np.zeros(nb_cluster)
#for j in range(nb_cluster):
#    count_total_u[j] = np.count_nonzero(class_Nu == j)
#    
#count_total_i = np.zeros(nb_cluster)
#for j in range(nb_cluster):
#    count_total_i[j] = np.count_nonzero(class_Ni == j)
    

def cluster_analiysis_1(liste,dataframe,mode):
    result = []
    list_result = []    
    
    for i in range(len(liste)):
        #mode : for filtering dataframe by a specific atribute
        if "users_age" in mode:
            result.append(dataframe[(dataframe['Age'] >= liste[i][0]) & (dataframe['Age'] < liste[i][1]) ].index.values.tolist())
        if "users_gender" in mode:
            result.append(dataframe[dataframe['Gender'] == liste[i] ].index.values.tolist())
        if "users_occupation" in mode:
            result.append(dataframe[dataframe['Occupation'] == liste[i] ].index.values.tolist())
        if "movies_genre" in mode:
            result.append(dataframe[dataframe['genres'].str.contains(liste[i])].index.values.tolist())
        
        new_result = [x+1 for x in result[i]]
        list_result.append(new_result)
        
    path_list = "data/data2/liste_"+ mode +".pkl"
    with open(path_list, 'wb') as fp:
        pickle.dump(list_result, fp)  
    return list_result

    
movies = pd.read_csv("data/data2/movies_pd.csv", sep="$")
list_genre = ["Action","Adventure","Animation",	"Children's",	"Comedy",	"Crime",	"Documentary",
    "Drama",	"Fantasy",	"Film-Noir",	"Horror",	"Musical",	"Mystery",	"Romance",	"Sci-Fi",	"Thriller",
	"War",	"Western"]
#
#movie_genre(list_genre, Ni_embed,movies)

##
#users = pd.read_table("data/data2/users.dat", sep="::", names= ['UserID','Gender','Age','Occupation','Zip-code'])
users = pd.read_csv("data/data2/users.csv", sep=",")
#users_age = users[users['Age'] < 20 ].index.values

#bokeh(users, Nu_embed, class_Nu)
#
#find_avg_age(nb_cluster,Nu_embed)

list_gender = ["F", "M"]
#fing_gender_major(nb_cluster, Nu_embed)
#user_gender(list_gender, Nu_embed, users, class_Nu, nb_cluster,count_total_u)

##   *  1:  "Under 18"
##	* 18:  "18-24"
##	* 25:  "25-34"
##	* 35:  "35-44"
##	* 45:  "45-49"
##	* 50:  "50-55"
##	* 56:  "56+"

list_age = [[0,18],[18,24],[25,34],[35,44],[45,49],[50,55],[56,100]]
#user_age(list_age, Nu_embed,users, class_Nu, nb_cluster,count_total_u)


#*  0:  "other" or not specified
#	*  1:  "academic/educator"
#	*  2:  "artist"
#	*  3:  "clerical/admin"
#	*  4:  "college/grad student"
#	*  5:  "customer service"
#	*  6:  "doctor/health care"
#	*  7:  "executive/managerial"
#	*  8:  "farmer"
#	*  9:  "homemaker"
#	* 10:  "K-12 student"
#	* 11:  "lawyer"
#	* 12:  "programmer"
#	* 13:  "retired"
#	* 14:  "sales/marketing"
#	* 15:  "scientist"
#	* 16:  "self-employed"
#	* 17:  "technician/engineer"
#	* 18:  "tradesman/craftsman"
#	* 19:  "unemployed"
#	* 20:  "writer"

list_occupation = list(range(21))

#users_age
#cluster_analiysis(list_age, Nu_embed, users, class_Nu, nb_cluster, count_total_u, "sub_users_age")
#users_gender
#cluster_analiysis(list_gender, Nu_embed, users, class_Nu, nb_cluster, count_total_u, "sub_users_gender")
#users_occupation
#cluster_analiysis(list_occupation, Nu_embed, users, class_Nu, nb_cluster, count_total_u, "sub_users_occupation")
#cluster_analiysis_2(list_gender,list_age, Nu_embed, users, class_Nu, nb_cluster, count_total_u, "2")
#movies_genre
#cluster_analiysis(list_genre, Ni_embed, movies, class_Ni, nb_cluster, count_total_i, "sub_movies_genre")
cluster_analiysis_1(list_occupation,users,"users_occupation")

#def test:
#    f = 0
#    m = 0
#    for i in range(len(class_Nu)):
#        if class_Nu[i] == 18:
#            print(i)
#            print(users['Gender'][i])
#            if users['Gender'][i] == 'F':
#                f += 1
#            else:
#                m +=1
#    print(f,m)
