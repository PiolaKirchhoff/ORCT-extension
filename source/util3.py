# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:16:01 2019

@author: antoc
"""

# Util.py is built in order to manage trees topologies & dataset architecture

import pandas as pd
import numpy as np
import random
# Object to manage trees of depth 2

#dictionaries to deal with ancient nodes of leaf nodes
#Right


BF_in_NL_R = {8:[],9:[4],10:[2],11:[2,5],12:[1],13:[1,6],14:[1,3],15:[1,3,7]}
#Left
BF_in_NL_L = {8:[1,2,4],9:[1,2],10:[1,5],11:[1],12:[3,6],13:[3],14:[7],15:[]}

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
    if i==2:
        return []
    elif i==3:
        return [1]
def B_in_NL(model, i):
    if i==2:
        return [1]
    if i==3:
        return []


# function to manage the percentage of predictor variables per tree (Global regularization)
def lit(l):
    return 1 if 1 in l else 0

#function to get seeds values for random initialization
def generate_random_init(n_exp):
    init_data = list(np.random.randint(low=0,high=10000, size=n_exp))
    
    init_var =[]
    for i in range(0,n_exp):
        var_i = list(np.random.randint(low=0,high=10000, size= 9))
        init_var.append(var_i)
    return {'init_data': init_data, 'init_var': init_var }

# function to deal with random tree for random forest
def sampling_dataset(alpha,dataset,depth,init=[]):
    
    num_nodes = 2**(depth) - 1
    ini = init if init else list(np.random.randint(low=0,high=10000, size=num_nodes))
    
    x = dataset.iloc[:,:-1].copy()
    y = dataset.iloc[:,-1].copy()
    return {i+1:pd.concat([x.sample(frac = alpha, replace = False, random_state = ini[i] , axis = 1),y],axis = 1) for i in range(num_nodes)}