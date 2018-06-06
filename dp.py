# -*- coding: utf-8 -*-

""" this file is used for data processing """
import numpy as np
import pandas as pd
import csv
import pickle
import codecs
import re

def preprocessing(row):
    """ enlever les virgules dans le nom de films """
    line = ','.join(row)
    line1 = line.split("::")
    #re.sub
    name_with_comma = re.findall(r'"([^"]*)"', line)
    if name_with_comma != []:
        name_with_comma = ' '.join(''.join(name_with_comma).split(","))
        line1[1] = name_with_comma
        new_line = " ".join(line1)
        
        return new_line
    else:
        return line 

def clean_movies(file_old,file_clean):
    """ écrire les données clean dans un nouveau fichier """
    lines = list()
    #encoding="utf8"
    with open(file_old,encoding="utf8") as f: # 'movies_new.csv'
      reader = csv.reader(f,delimiter='::')      
      for row in reader:
          #row = preprocessing(row)
          lines.append(row)
    with open(file_clean, "w", encoding="utf8", newline='') as csv_file: #"movies_cleaned_new.csv"
            writer = csv.writer(csv_file)
            for line in lines:
                writer.writerow([line])

def get_data(file):
    types_of_encoding = ["utf8","cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open(file,encoding = encoding_type, errors = 'replace') as f:
            movies = [tuple(line) for line in csv.reader(f)]
    for i in range(len(movies)):
        movies[i] = movies[i][0].split("::")
    return movies

    
def get_hash_index(file,label):
    hash_index = dict()
    movies_pd = pd.read_table(file, sep='::',names = label)
    index = movies_pd.index.values
    for i in index:
        hash_index[movies_pd[label[0]][i]] = i+1  
    return hash_index,movies_pd



""" Pour traiter df pd """
def movies_change_id(file1,col,file2):
    hash_index = dict()
    movies_pd = pd.read_table(file1, sep='::',names = col)
    for i,row in movies_pd.iterrows():
        hash_index[row[col[0]]] = i+1
        movies_pd[col[0]][i] = i+1
    movies_pd.to_csv(file2,sep='$',columns = col,index = False)
    return hash_index,movies_pd

def ratings_change_id(file,hash_index,file2):
    ratings  = np.genfromtxt(file,delimiter='::',dtype=(int,int,int,int))
    for r in ratings:
        r[1] = hash_index[r[1]]
    np.save(file2,ratings)
    return ratings

def get_ratings_pd():
    dtype = [('UserID','int'), ('MovieID','int'), ('Rating','int'), ('Timestamp','int')]
    values = np.load("data/data2/ratings.npy")
    values.astype(dtype)
    index = [i for i in range(1, len(values)+1)]
    ratings = pd.DataFrame(values, index=index)
    ratings.rename(columns = {'f0':'UserID','f1':'MovieID','f2':'Rating','f3':'Timestamp'}, inplace=True)
    return ratings

        

#hash_index,movies = movies_change_id("data/data2/movies.dat",['movieId', 'title', 'genres'],"data/data2/movies_pd.csv")
#ratings = ratings_change_id("data/data2/ratings.dat",hash_index,"data/data2/ratings.npy")

def sub_sample():
    users = pd.read_table("data/data2/users.dat",sep='::', names = ['UserId','Gender','Age','Occupation','ZipCode'])
    users_sub = users.sample(600)
    users_sub = users_sub.sort_values(by=['UserId']).reset_index(drop=True)
    index_users_dict = dict()
    for i,row in users_sub.iterrows():
            index_users_dict[row['UserId']] = i+1
            users_sub['UserId'][i] = i+1
        
    movies = pd.read_table("data/data2/movies.dat", sep='::',names = ['movieId', 'title', 'genres'])       
    movies_sub = movies.sample(300)
    movies_sub = movies_sub.sort_values(by = ['movieId']).reset_index(drop=True) 
    index_movies_dict = dict()
    for i,row in movies_sub.iterrows():
            index_movies_dict[row['movieId']] = i+1
            movies_sub['movieId'][i] = i+1 
    movies_sub.to_csv("data/data2/sub_movies.csv",columns = ['movieId', 'title', 'genres'],index = False)
    users_sub.to_csv("data/data2/sub_users.csv",columns = ['UserId','Gender','Age','Occupation','ZipCode'],index = False)   
    
    userratings = np.load("data/data2/ratings.npy")
    ratings = userratings[:1]
    for i, r in enumerate(userratings):
        if (r[0] in index_users_dict.keys()) and (r[1] in index_movies_dict.keys()):
            r[0] = index_users_dict[r[0]]
            r[1] = index_movies_dict[r[1]]
            ratings = np.append(ratings,[r],axis = 0)
            print("update",i,"r = ", r)
    ratings = np.delete(ratings,0,0)
    np.save("data/data2/sub_ratings.npy",ratings)  

def contatenate_list_users():
    with open ("data/data2/liste_users_gender.pkl", 'rb') as fp:
        list_users_gender = pickle.load(fp)
    with open ("data/data2/liste_users_age.pkl", 'rb') as fp:
        list_users_age = pickle.load(fp)
    with open ("data/data2/liste_users_occupation.pkl", 'rb') as fp:
        list_users_occupation = pickle.load(fp)
        
    for e in list_users_age:
        list_users_gender.append(e)
    for e in list_users_occupation:
        list_users_gender.append(e)
        
    with open ("data/data2/liste_users.pkl", 'wb') as fp:
         pickle.dump(list_users_gender,fp)



 
