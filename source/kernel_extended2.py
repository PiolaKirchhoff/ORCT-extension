# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:48:50 2019

@author: antoc
"""

import numpy as np
import pandas as pd
import math
from math import log
from math import exp
from functools import reduce
import operator
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from source.util import *
from source.maths1 import *
from source.sorct2_ex import *

class extended_kernel:
    """ The class manage expansion of kernel space for Sparsity in Optimal Randomized Classification Trees (SORCT) of Blanquero et Al. 2018
    and its extensions made by us. The depth of the tree is set to 2.

    Parameters:
        df (pandas.dataframe)       - New data
        test (pandas.dataframe)     - New test data
        N (int)                     - Number of iterations of kernel expansion 
        I_k (method)                - Method to manage the dictionary defined above to deal with Pyomo sets    
        init(dictionary)            - Initialization values
        type_reg(string)            - Name of model to choose
        reg_term(float)             - value for the regularization term
        ini(list)                   - Alternative for initialization values
    """
    def __init__(self,df,test,N=3,init={},type_reg="simple",reg_term = 0,ini = []):
        """ The constructor initializes dictionaries and methods to deal with decision tress topologies.
            There is a preprocessing phase to deal with dataset.By default the dataset is assumed as follows:
            - The column of the labels must be named "Classes"
            - The values that labels asummed are encoded by integers
        
        """
        
        self.data_set = df
        
        # number of iterations
        self.iterations = N
        # setting of a list to save all the models after updating phase
        self.learners = []
        # setting of a list to store all the dataset generated by updating phase
        self.dfs = [df]
        
        # test set in order to have test accuracy
        self.test = test
        
        # I_in_k is a dictionary built by method in util to deal with pyomo syntax the objective function: the key of dictionary is the class labels the values are the indeces of instances belonging to that class
        self.I_in_k = inst_class(df)
        # list of classes in our dataset
        self.classes = list(self.I_in_k.keys())
        
        # I_k is a method that given a class label returns the indeces of instances belonging to that class
        # this if statement to generalize models of DT of depth at most 2 for any number of classes
        if len(self.classes)== 3:
                
            def I_k(model,i):
                if i==0:
                    return self.I_in_k[0]
                elif i==1:
                    return self.I_in_k[1]
                elif i==2:
                    return self.I_in_k[2]
        else:
            def I_k(model,i):
                if i==0:
                    return self.I_in_k[0]
                elif i==1:
                    return self.I_in_k[1]
                elif i==2:
                    return self.I_in_k[2]
                elif i==3:
                    return self.I_in_k[3]
        
        self.I_k = I_k  
        
        # generate random_init is a method defined in util whcih generates a list of random seeds
        self.init = init if init else generate_random_init(N)
        # list of class labels
        self.classes = self.data_set['Classes'].unique().tolist()
        
        # typre_reg is a string that express which model you want to use
        self.type_reg = type_reg
        # regularization term
        self.reg_term = reg_term
        
        #  alternative initialization: for the experiments of comparison with normal model with no expansion i use this one
        self.initialization = ini
        
        # this dictionary is fundamental to list all the couples of feature that update of kernel expansion create
        self.d = {}
        # number of iterations made by the class
        self.it = 0
        
    def update(self, it = 1):
        
        # By default only one step for updating is made
        for i in range(it):
            
            
            # initialization term
            ini =[]
            
            # if statement to deal with the two ways of initialize start for variables
            if self.initialization:
                ini = self.initialization
            else:
                np.random.seed(self.init['init_var'][self.it][0])
                init_a = np.random.uniform(low=-1.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][1])
                init_mu = np.random.uniform(low=-1.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][2])
                init_C = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][3])
                init_P = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][4])
                init_p = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][5])
                init_beta = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][6])
                init_am = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][7])
                init_ap = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][8])
                init_z = np.random.uniform(low=0.0, high=1.0, size=None)
                
                ini = [init_a,init_mu,init_C,init_P,init_p,init_am,init_ap,init_beta,init_z]
            
            # LOOK AT sorct2_ex.py file for details
            # Initialization of class SORCT for building a model to store in the mother class
            w_l = SORCT_2(self.data_set,self.test,self.I_in_k,self.I_k,d = self.d)
            # Setting initialization for starting
            w_l.set_init(ini)
            # Creating the model with pyomo syntax
            w_l.createModel()
            # Uploading the type of model
            w_l.charge_of(self.type_reg,self.reg_term)
            
            
            # Solving phase: in general i use two times
            try:
                w_l.solve()
            except Exception as e:
                print (str(e))
            try:
                w_l.solve()
            except Exception as e:
                print (str(e))   
            # Store the results in a dictionary in w_l
            w_l.extraction_va()
            
            # Building phase for the new dataset with the extension criterion 
            new_x = w_l.update_rule1()
            
            # Calculate the new regularization term of the updated dataframe
            new_reg = 1/(3*(len(new_x.columns)))
            self.reg_term = new_reg
            
            # Building phase for the new testing set with the extension criterion
            new_x_test = w_l.test_generation()
            y_test = self.test.iloc[:,-1]
            self.test = pd.concat([new_x_test, y_test], axis = 1, join_axes=[new_x_test.index])
            
            # Update the dictionary which stores the feature already present in our analysis
            self.d = w_l.d
            # column with the labels to be concatenated to the new dataframe of features 
            y = self.data_set.iloc[:,-1]
            self.data_set = pd.concat([new_x, y], axis = 1, join_axes=[new_x.index])
            
            # This three passages are made to solve a bug of sklearn preprocessing for normalization
            self.data_set = self.data_set.replace(1.0000000000000002, 1)
            self.data_set = self.data_set.replace(1.0000000000000004, 1)
            self.data_set = self.data_set.replace(1.0000000000000009, 1)

            
            # self.dic is a dictionary fundamental for conversion of the new added features to the old name for example: after 1 updating phasing we have a new 9 th features that in reality express 0-0 
            # if we are in the first iterations it simply create the dictionary and append the new dataset tp the list of dataframes
            # otherwise it make conversion of names of the new features with old names
            if self.it == 0:
                self.dic = {i: self.data_set.columns[i] for i in range(len(self.data_set.columns)-1)}
                self.dfs.append(self.data_set)
            else:
                #l will be the list with the name of all the columns of the new dataset
                l = []
                #print(self.data_set.columns)
                for i in self.data_set.columns[:-1]:
                    s = ''
                    for j in i:
                        if j!='-':
                            s+= self.dic[int(j)]
                    l.append(s)
                l.append('Classes')
                self.data_set.columns = l
                self.dic = {i: self.data_set.columns[i] for i in range(len(self.data_set.columns)-1)}
                self.dfs.append(self.data_set)
            
            # append the model to the list of the models
            self.learners.append(w_l)
            # update the iterations    
            self.it += 1    
                
        return True
    
   
    
    def solve(self):
        

        for i in range(self.iterations):
            
            # initialization term
            ini =[]
            
            # if statement to deal with the two ways of initialize start for variables
            if self.initialization:
                ini = self.initialization
            else:
                np.random.seed(self.init['init_var'][self.it][0])
                init_a = np.random.uniform(low=-1.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][1])
                init_mu = np.random.uniform(low=-1.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][2])
                init_C = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][3])
                init_P = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][4])
                init_p = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][5])
                init_beta = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][6])
                init_am = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][7])
                init_ap = np.random.uniform(low=0.0, high=1.0, size=None)
                np.random.seed(self.init['init_var'][self.it][8])
                init_z = np.random.uniform(low=0.0, high=1.0, size=None)
                
                ini = [init_a,init_mu,init_C,init_P,init_p,init_am,init_ap,init_beta,init_z]
            
            # LOOK AT sorct2_ex.py file for details
            # Initialization of class SORCT for building a model to store in the mother class
            w_l = SORCT_2(self.data_set,self.test,self.I_in_k,self.I_k,d = self.d)
            # Setting initialization for starting
            w_l.set_init(ini)
            # Creating the model with pyomo syntax
            w_l.createModel()
            # Uploading the type of model
            w_l.charge_of(self.type_reg,self.reg_term)
            
            
            # Solving phase: in general i use two times
            try:
                w_l.solve()
            except Exception as e:
                print (str(e))
            try:
                w_l.solve()
            except Exception as e:
                print (str(e))   
            # Store the results in a dictionary in w_l
            w_l.extraction_va()
            
            # Building phase for the new dataset with the extension criterion 
            new_x = w_l.update_rule1()
            
            # Calculate the new regularization term of the updated dataframe
            new_reg = 1/(3*(len(new_x.columns)))
            self.reg_term = new_reg
            
            # Building phase for the new testing set with the extension criterion
            new_x_test = w_l.test_generation()
            y_test = self.test.iloc[:,-1]
            self.test = pd.concat([new_x_test, y_test], axis = 1, join_axes=[new_x_test.index])
            
            # Update the dictionary which stores the feature already present in our analysis
            self.d = w_l.d
            # column with the labels to be concatenated to the new dataframe of features 
            y = self.data_set.iloc[:,-1]
            self.data_set = pd.concat([new_x, y], axis = 1, join_axes=[new_x.index])
            
            # This three passages are made to solve a bug of sklearn preprocessing for normalization
            self.data_set = self.data_set.replace(1.0000000000000002, 1)
            self.data_set = self.data_set.replace(1.0000000000000004, 1)
            self.data_set = self.data_set.replace(1.0000000000000009, 1)

            
            # self.dic is a dictionary fundamental for conversion of the new added features to the old name for example: after 1 updating phasing we have a new 9 th features that in reality express 0-0 
            # if we are in the first iterations it simply create the dictionary and append the new dataset tp the list of dataframes
            # otherwise it make conversion of names of the new features with old names
            if self.it == 0:
                self.dic = {i: self.data_set.columns[i] for i in range(len(self.data_set.columns)-1)}
                self.dfs.append(self.data_set)
            else:
                #l will be the list with the name of all the columns of the new dataset
                l = []
                #print(self.data_set.columns)
                for i in self.data_set.columns[:-1]:
                    s = ''
                    for j in i:
                        if j!='-':
                            s+= self.dic[int(j)]
                    l.append(s)
                l.append('Classes')
                self.data_set.columns = l
                self.dic = {i: self.data_set.columns[i] for i in range(len(self.data_set.columns)-1)}
                self.dfs.append(self.data_set)
            
            # append the model to the list of the models
            self.learners.append(w_l)
            # update the iterations    
            self.it += 1 
            
        return True
    
    # this method let us to make prediction on a single instance  taking as reference the last model added to the list    
    def comp_label(self,x):
        """ This method predictes label of a single instance
        
        Param:
            x (array) - single instance
        Return:
            result (integer) - label of class
        
        """    

        return self.learners[-1].comp_label(x)

    def predicted_lab(self,X_test):
        """ This method predictes label of a several instances
        
        Param:
            X_test (pandas.dataframe) - test set of instances
        
        Return:
            label (array) - labels of class
        
        """
        label = []
        for i in range(0,len(X_test)):
            label.append(self.comp_label(list(X_test.iloc[i])))
        return label
    # this method let us to make prediction on the test set passed by the constructor taking as reference the last model added to the list
    def predicted(self):
        """ This method predictes labels of the instances stored in self.test
        
        
        Return:
            label (array) - labels of class
        
        """
        return self.learners[-1].predicted()    

    #Calculate the accuracy out of sample
    def accuracy(self,y,y_pred):
        """ This method return the accuracy 
        
        Params:
            y (array)       - test set actual labels
            y_pred (array)  - test set predicted labels
        
        Return:
            result (float) - percentage of accuracy
        
        """
        l = [1 if list(y)[i]==list(y_pred)[i] else 0 for i in range(0,len(y))]
        return sum(l)/len(y)
    

    
        
        