#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:48:29 2018

@author: astricot
"""
#import clustering as cls
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

#cls.write_tsne()


#f1 = 'data/data2/Nu_embed.npy'
#Nu_embed = np.load(f1)
##class_Nu = cls.clustering(Nu_embed,30)
#plt.figure()
#plt.scatter(Nu_embed[:,0], Nu_embed[:,1], marker='s', s=20)
#users = pd.read_table("data/data2/users.dat",sep='::', names = ['UserId','Gender','Age','Occupation','ZipCode'])
#users_ind = users[users['Age']>50].index.values
#Nu_emb_age =  Nu_embed.take(users_ind,axis=0)
#plt.scatter(Nu_emb_age[:,0], Nu_emb_age[:,1], c='red', marker='.', s=20)
#plt.savefig("result/data2/agesup50.pdf")
#plt.show()

#list_genre=["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
#f2 = 'data/data2/Ni_embed.npy'
#Ni_embed = np.load(f2)
#class_Ni = cls.clustering(Ni_embed,30)
#movies = pd.read_csv("data/data2/movies_pd.csv",sep='$')
#cls.movie_genre(movies,class_Ni,list_genre,Ni_embed)



#users = pd.read_table("data/data2/users.dat",sep='::', names = ['UserId','Gender','Age','Occupation','ZipCode'])
#users.to_csv("data/data2/users.csv",sep=',',columns = ['UserId','Gender','Age','Occupation','ZipCode'],index = False)
#usersF = users[users['Gender']=='F']


