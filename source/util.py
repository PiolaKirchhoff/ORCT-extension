# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:16:01 2019

@author: antoc
"""

# Util.py is built in order to manage trees topologies & dataset architecture

import pandas as pd

# Object to manage trees of depth 2

#dictionaries to deal with ancient nodes of leaf nodes
#Right
BF_in_NL_R = {4:[],5:[2],6:[1],7:[1,3]}
#Left
BF_in_NL_L = {4:[1,2],5:[1],6:[3],7:[]}

# By default the dataset is assumed as follows:
# - The column of the labels must be named "Classes"
# - The values that labels asummed are encoded by integer

#function to manage misclassification: return an dictionary of misclassificstion costs 
def cost(dataset):
   #The name of the classes K
   classes = dataset['Classes'].unique().tolist()
   #Encoding
   classes_en = [i for i in range(len(classes))] 
   
   return {(i,j): 0.5 if i != j else 0 for i in classes_en for j in classes_en}

#function to deal with indeces of instances among classes
def inst_class(dataset):
    #The name of the classes K
    classes = dataset['Classes'].unique().tolist()
    return {i : list(dataset[dataset['Classes']== i].index) for i in range(len(classes))}

#function to deal with training set 
# the input dataset must be a dataframe in pandas with all the column except for labels column
def my_train(dataset):
    index_instances = list(dataset.index)
    index_features = list(range(0,len(dataset.columns)-1))
    
    return  {(i,j): dataset.loc[i][j] for i in index_instances for j in index_features}

#function to manage ancient nodes of leaf nodes
def B_in_NR(model, i):
    if i==4:
        return []
    elif i==5:
        return [2]
    elif i==6:
        return [1]
    elif i==7:
        return [1,3]
def B_in_NL(model, i):
    if i==4:
        return [1,2]
    elif i==5:
        return [1]
    elif i==6:
        return [3]
    elif i==7:
        return []