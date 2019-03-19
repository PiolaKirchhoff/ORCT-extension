# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:10:46 2019

@author: antoc
"""
import math
import numpy as np

# maths.py has math functionalities

# The default value is 512 as suggested in Blanquero et Al.
#Sigmoid function
def my_sigmoid(a,x,mu,scale=512):
    l = len(x)
    val = (sum([a[i]*x   for i, x in enumerate(x)]) / l) - mu 
    
    return 1 / (1 + math.exp(-scale*val))

# An easy way to manage product within elements of an iterable object
def multiply_numpy(iterable):
    return np.prod(np.array(iterable))

# Calculate the probability of an individual falling into a given leaf node:
def Prob(model,index_features,var,x, leaf_idx):
    left = [my_sigmoid(list(var['a']['a['+str(i)+','+str(tl)+']'] for i in index_features),x,var['mu']['mu['+str(tl)+']']) for tl in model.N_L_L[leaf_idx] ]
    right = [1-my_sigmoid(list(var['a']['a['+str(i)+','+str(tr)+']'] for i in index_features),x,var['mu']['mu['+str(tr)+']']) for tr in model.N_L_R[leaf_idx] ]
    return multiply_numpy(left)*multiply_numpy(right)
