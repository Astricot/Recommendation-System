#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:45:27 2018

@author: astricot
"""
import numpy as np

from sklearn import cluster,datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tSNE_Nu(file1,file2):
    N = np.load(file1)
    N_embed = TSNE(n_components=2).fit_transform(N)
    np.save(file2, N_embed)

def tSNE_Ni(file1,file2):
    N = np.load(file1)
    N_embed = TSNE(n_components=2).fit_transform(N.transpose())
    np.save(file2, N_embed)
    
def clustering(N_embed,nb):
    c = cluster.AgglomerativeClustering(nb)
    c.fit(N_embed)
    c_class = c.labels_
    return c_class

def tSNE_iter(file):
    for i in range(5,61,5):
        print(i)
        N = np.load(file)
        N_embed = TSNE(n_components=2,perplexity=i,n_iter=5000).fit_transform(N)
        pathnp = "data/data2/embed/perplexity" + str(i) + ".npy"
        np.save(pathnp,N_embed)
        pathpdf = "data/data2/embed/perplexity" + str(i) + ".pdf"
        plt.figure()
        plt.scatter(N_embed[:,0], N_embed[:,1], marker='.', s=20)
        plt.savefig(pathpdf)

def write_tsne():
    f1 = 'data/data2/Nu.npy'
    f2 = 'data/data2/Ni.npy'
    f3 = 'data/data2/Nu_embed.npy'
    f4 = 'data/data2/Ni_embed.npy'
    tSNE_Nu(f1,f3)
    tSNE_Ni(f2,f4)

def write_tsne_z12():
    f1 = "result/data2/test_dim/Nu_z12.npy"
    f2 = "result/data2/test_dim/Ni_z12.npy"
    f3 = "result/data2/test_dim/Nu_embed_z12.npy"
    f4 = "result/data2/test_dim/Ni_embed_z12.npy"
    tSNE_Nu(f1,f3)
    tSNE_Ni(f2,f4)
    
def movie_genre(movies,class_Ni,list_genre,Ni_embed):
    """ tracer selon genre de film """
    movies_index = []
    for i in range(len(list_genre)):
        movies_index.append(movies[movies['genres'].str.contains(list_genre[i])].index.values)
        Ni_emb_mov =  Ni_embed.take(movies_index[i],axis=0)
        nom = "result/data2/movies_genre/" + list_genre[i] + ".pdf"
        plt.figure()
        plt.scatter(Ni_embed[:,0], Ni_embed[:,1],c=class_Ni, marker='s', s=20)
        plt.scatter(Ni_emb_mov[:,0], Ni_emb_mov[:,1],c='red', marker='o', s=10)
        plt.savefig(nom)
        plt.show()
        plt.close()

#tSNE_iter("data/data2/Nu.npy")
#write_tsne_z12()
#tSNE_iter('data/data2/Nu.npy')