# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:57:18 2019

@author: antoc
"""
import numpy as np
import pandas as pd
import math
from functools import reduce
import operator
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from source.util import *
from source.maths_random import *
from source.liana import *


class Jungle:
    """ The class manage random forest embedding for Optimal Randomized Classification Trees (liana).
        I call it "Random Jungle".
        The depth of each liana is set to 2 by default but can be changed.

    Parameters:
        df (pandas.dataframe)       - New data
        N (integer)                 - Number of times
        depth(integer)              - depth of each tree generated by random jungle
        percent(float)              - percentage of features involved in each decision node of each tree
        init(dictionary)            - Initialization values for the variables and the bootstrapping: it requires two values : 'init_data' for sampling and 'init_var' for initialization
        type_reg(string)            - Name of model to choose
        reg_term(float)             - value for the regularization term    
    """
    
    
    def __init__(self, df, N = 100, percent=0.5, init={},type_reg="simple",reg_term = 0):
        """ The constructor initializes number of times, dataframes and initialization values.
        
        """
        
        # setting of dataframe passe by constructor
        self.df = df
        # setting number of iterations for method solve()
        self.N = N
        # I_in_k is a dictionary passed by the constructor to deal with pyomo syntax the objective function: the key of dictionary is the class labels the values are the indeces of instances belonging to that class
        self.I_in_k = inst_class(df)        
        # list of classes in our dataset
        self.classes = list(self.I_in_k.keys())
        
        # depth of decision trees to create
        self.depth = 2 if len(self.classes) >= 3 else 1
        # percentage of features involved in each decision node of each tree
        self.percent = percent
        
        # typre_reg is a string that express which model you want to use
        self.type_reg = type_reg
        # regularization term
        self.reg_term = reg_term
        
        self.columns_names = list(df)
        
        
        # generate random_init is a method defined in util which generates a list of random seeds or set the value passed by constructor
        self.init = init if init else generate_random_init(self.N)
        self.list_liana = []
        self.indeces_out_of_bag = {}
        self.it = 0
    
    def solve(self):
        """This method develops the following procedure:
            A bootstrap sample LB is selected from df, and a tree grown using LB.
            This is repeated N times giving tree classifiers ORCT1, ORCT2,..., ORCTN.
        
        Return:
            result (boolean) - solve function result
        
        """
        print("Welcome to the jungle!\n")
        for i in range(self.N):
            
            # boostrap phase of dataset
            data = self.df.sample(n = self.df.shape[0], random_state = self.init['init_data'][self.it], replace = True)
            # indeces of instances sampled by the bootsrap: this list is useful in order to calculate the out of bag accuracy defined by L. Breiman
            bad_df = self.df.index.isin(list(data.index))
            self.indeces_out_of_bag[self.it] = list(self.df[~bad_df].index)
            data = data.reset_index(drop=True)
            # I_in_k is a dictionary built by method in util to deal with pyomo syntax the objective function: the key of dictionary is the class labels the values are the indeces of instances belonging to that class
            I_in_k = inst_class(data)
            
            # I_k is a method that given a class label returns the indeces of instances belonging to that class
            # this if statement is in order to generalize models of DT of depth at most 2 for any number of classes
            if len(self.classes)== 3:
                
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
                    elif i==2:
                        return I_in_k[2]
            elif len(self.classes)==2:
                
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
            else:
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
                    elif i==2:
                        return I_in_k[2]
                    elif i==3:
                        return I_in_k[3]
                    
            # INITIALIZATION FOR VARIABLES
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

    
            ini = [init_a,init_mu,init_C,init_P,init_p,init_beta,init_am,init_ap,init_z]
            # LOOK AT liana.py file for details
            # Initialization of class random_tree for building a model to store in the mother class
            liana = random_tree(data,self.percent,I_in_k,I_k,depth =self.depth)
            # Setting initialization for starting
            liana.set_init(ini)
            # Creating the model with pyomo syntax
            liana.createModel()
            # Uploading the type of model
            liana.charge_of(self.type_reg,self.reg_term)
    
            try:
                liana.solve()
            except Exception as e:
                print (str(e))
                self.it+=1
                continue
            try:
                liana.solve()
            except Exception as e:
                print (str(e))
                self.it+=1
                continue
            liana.extraction_va()
            self.list_liana.append(liana)
            self.it+=1
        print("\n")    
        print("""Watch it bring you to your shun n-n-n-n-n-n-n-n knees, knees
In the jungle welcome to the jungle
Watch it bring you to you
Its gonna bring you down, ha!""")
        
        return True
    
    def update(self,it = 1):
        """This method develops the following procedure:
            A bootstrap sample LB is selected from df, and a tree grown using LB.
            This is repeated N times giving tree classifiers ORCT1, ORCT2,..., ORCTN.
        
        Return:
            result (boolean) - solve function result
        
        """
        print("Welcome to the jungle!\n")
        for i in range(it):
            
            # boostrap phase of dataset
            data = self.df.sample(n = self.df.shape[0], random_state = self.init['init_data'][self.it], replace = True)
            # indeces of instances sampled by the bootsrap: this list is useful in order to calculate the out of bag accuracy defined by L. Breiman
            bad_df = self.df.index.isin(list(data.index))
            self.indeces_out_of_bag[self.it] = list(self.df[~bad_df].index)
            data = data.reset_index(drop=True)
            # I_in_k is a dictionary built by method in util to deal with pyomo syntax the objective function: the key of dictionary is the class labels the values are the indeces of instances belonging to that class
            I_in_k = inst_class(data)
            
            # I_k is a method that given a class label returns the indeces of instances belonging to that class
            # this if statement is in order to generalize models of DT of depth at most 2 for any number of classes
            if len(self.classes)== 3:
                
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
                    elif i==2:
                        return I_in_k[2]
            elif len(self.classes)==2:
                
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
            else:
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
                    elif i==2:
                        return I_in_k[2]
                    elif i==3:
                        return I_in_k[3]
                    
            # INITIALIZATION FOR VARIABLES
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

    
            ini = [init_a,init_mu,init_C,init_P,init_p,init_beta,init_am,init_ap,init_z]
            # LOOK AT liana.py file for details
            # Initialization of class random_tree for building a model to store in the mother class            
            liana = random_tree(data,self.percent,I_in_k,I_k,depth =self.depth)
            # Setting initialization for starting
            liana.set_init(ini)
            # Creating the model with pyomo syntax
            liana.createModel()
            # Uploading the type of model
            liana.charge_of(self.type_reg,self.reg_term)
    
            try:
                liana.solve()
            except Exception as e:
                print (str(e))
                self.it+=1
                continue
            try:
                liana.solve()
            except Exception as e:
                print (str(e))
                self.it+=1
                continue
            liana.extraction_va()
            self.list_liana.append(liana)
            self.it+=1

        
        return True
        
    
    def out_of_bag(self):
        """This method develops the procedure invented by L. Breiman for analysing the perfomance of accuracy on a dataset 
            Since a bootstrap sampling LB is made from df at each step, this methods store the indeces of instances not used in some boostrap step and ask to them to predict the labels for those indeces 
            In this way it is possible to have a robust result and also use the whole dataset.
        
        Return:
            accuracy (float) - percentage of accuracy on the dataset loaded in constructor
        
        """
        predictions = []
        for i in range(self.df.shape[0]):
            
            index_bagged = self.find_index(i)

            if index_bagged:
                pr = [self.list_liana[j].predicted_lab(pd.DataFrame(self.df[self.columns_names[:-1]].iloc[i,:]).T)[0] for j in index_bagged]
                pr_array = np.array(pr)
                counts = np.bincount(pr_array)

                predictions.append(np.argmax(counts))
            else:
                predictions.append('No')
        
        s = []
        for i in range(0,len(self.df['Classes'])):
            if list(self.df['Classes'])[i]==predictions[i]:
                s.append(1)
            elif predictions[i]!='No':
                s.append(0)
            
        return sum(s)/len(s)  
    
    def find_index(self,index):
        """This method is fundamental in out_of_bag() method. Given an index of a instance it return the list of liane in which it was not bootstraped  .
            
        
        Return:
            list_ind (list) - list of liane not trained with the instance of that index
        
        """
        list_ind = []
        for i in range(len(self.list_liana)):
            if index in self.indeces_out_of_bag[i]:
                list_ind.append(i)
                
        return list_ind
    
    def predict(self,X_test):
        """ This method predictes label of a several instances and store the results
             in a list.
        
        Param:
            X_test (pandas.datafram) - test set of instances
        
        Return:
            result (boolean) - solve function result
        
        """       
        pr = [self.list_liana[i].predicted_lab(X_test) for i in range(len(self.list_liana))]
        pr_array = np.array(pr)
        
        predictions = []
        for i in range(X_test.shape[0]):
            counts = np.bincount(pr_array[:,i])
            predictions.append(np.argmax(counts))
        
        self.predictions = predictions
        
        if predictions:
            return predictions
        else:
            return False
     
    def accuracy(self,y):
        """ This method return the accuracy given the predicted labels stored in the class.
        
        Params:
            y (array)       - test set actual labels
        
        Return:
            result (float) - percentage of accuracy
        
        """
        l = [1 if list(y)[i]==self.predictions[i] else 0 for i in range(0,len(y))]
        return sum(l)/len(y)
    
    def set_n(self,new):
        """ This method set the number of iterations
        
        Param:
            new    (integer) - number of iterations
        
        Return:
            result (boolean) - solve function result
        
        """     
        self.N = new
        
        return True
        