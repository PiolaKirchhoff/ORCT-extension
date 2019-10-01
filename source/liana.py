# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:00:18 2019

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


class random_tree:
    
    def __init__(self,dataset,percent,I_in_k,I_k,depth=2,init=[]):
        """ The constructor initializes dictionaries and methods to deal with decision tress topologies.
            There is a preprocessing phase to deal with dataset.By default the dataset is assumed as follows:
            - The column of the labels must be named "Classes"
            - The values that labels asummed are encoded by integers
        
        """
        # B_in_NL & B_in_NR,B_in_NL1 & B_in_NR1  are defined in util: dictionaries to deal with ancient nodes of leaf nodes
        self.B_in_NL = B_in_NL if depth==2 else B_in_NL1
        self.B_in_NR = B_in_NR if depth==2 else B_in_NR1
        # BF_in_NL_L & BF_in_NL_R,BF_in_NL_L1 & BF_in_NL_R1 are defined in util: they are functions to manage ancient nodes of leaf nodes
        self.BF_in_NL_R = BF_in_NL_R if depth==2 else BF_in_NL_R1
        self.BF_in_NL_L = BF_in_NL_L if depth==2 else BF_in_NL_L1
        # list of decison nodes
        self.BR_NODES = list([i+1 for i in range(2**(depth) - 1)] )
        
        # generate random_init is a method defined in util whcih generates a list of random seeds
        self.initi = list(np.random.randint(low=0,high=10000, size=len(self.BR_NODES)))
        # I_in_k is a dictionary passed by the constructor to deal with pyomo syntax the objective function: the key of dictionary is the class labels the values are the indeces of instances belonging to that class
        self.I_in_k = I_in_k
        # I_k is a method passed by the constructor to deal with pyomo syntax the objective function: given a class label the method returns the indeces of instances belonging to that class
        self.I_k = I_k
        # my_W is a dictionary, cost is a method defined in util1: it returns an dictionary of misclassification costs
        self.my_W = cost(dataset)

        self.percent = percent
        self.depth = depth
        self.dict_df = sampling_dataset(percent,dataset,self.depth,init=self.initi)
        self.index_features = list(range(0,len(self.dict_df[1].columns)-1))
        self.index_instances = list(dataset.index)
        self.classes = dataset['Classes'].unique().tolist()
        
        # This dictionary is useful in order to deal with sampling random subspace of features in each node: so the key is a triplet of label of node, index of instance and index of features, the value is the value of that feature for that instance in a given decision node
        self.my_x = {(tb,i,j): self.dict_df[tb].loc[i][j] for tb in self.BR_NODES for i in self.index_instances for j in self.index_features}
        
        # number of features
        self.number_f = len(self.dict_df[1].columns)-1

        
        # initilization values for variables of the model in case you don't use the proper method
        self.init_a = init[0] if len(init) > 0 else np.random.uniform(low=-1.0, high=1.0, size=None)
        self.init_mu = init[1] if len(init) > 0 else np.random.uniform(low=-1.0, high=1.0, size=None)
        self.init_C = init[2] if len(init) > 0 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_P = init[3] if len(init) > 0 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_am = init[5] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_ap = init[6] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_b = init[7] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_z = init[8] if len(init) > 5 else np.random.uniform(low=-1.0, high=1.0, size=None)
        
    def createModel(self):
        """ This method builds the skeleton of the Decision Tree through Pyomo syntax
        
        Return:
            result (boolean)    - createModel function result
        """
        #Model definition
        self.model = ConcreteModel() #ConcretModel()
        
        #Set definition
        self.model.I = Set(initialize=set(i for k in self.I_in_k for i in self.I_in_k[k]))
        self.model.K = Set(initialize=self.I_in_k.keys())
        self.model.I_k = Set(self.model.K,initialize=self.I_k)    
        self.model.f_s = Set(initialize=self.index_features)
        self.model.N_B = Set(initialize=set(i for k in self.BF_in_NL_R for i in self.BF_in_NL_R[k]))
        self.model.N_L = Set(initialize=self.BF_in_NL_R.keys())
        self.model.N_L_R = Set(self.model.N_L,initialize=self.B_in_NR)
        self.model.N_L_L = Set(self.model.N_L,initialize=self.B_in_NL)
        
        #Parameters definition
        self.model.l_inf = Param(initialize= 0,mutable=True)
        self.model.l_l1 = Param(initialize= 0,mutable=True)
        self.model.l_l2 = Param(initialize= 0,mutable=True)
        self.model.l_l0 = Param(initialize= 0,mutable=True)
        self.model.l_l0_inf = Param(initialize= 0,mutable=True)
        self.model.l_log = Param(initialize= 0,mutable=True)
        self.model.pr1 = Param(initialize= 0,mutable=True)
        self.model.pr2 = Param(initialize= 0,mutable=True)
        self.model.W = Param(self.model.K, self.model.K, within=NonNegativeReals, initialize=self.my_W)
        self.model.x = Param(self.model.N_B,self.model.I, self.model.f_s, within=PercentFraction, initialize=self.my_x)
        self.model.alpha = Param(initialize= 0,mutable=True)
        
        #Variables definition
        self.model.a = Var(self.model.f_s, self.model.N_B, within=Reals, bounds = (-1.0,1.0),initialize=self.init_a)
        #auxiliary variables for smooth version of regularization (local & global)
        self.model.a_minus = Var(self.model.f_s, self.model.N_B,  within = PercentFraction ,initialize=self.init_am)
        self.model.a_plus = Var(self.model.f_s, self.model.N_B,  within = PercentFraction,initialize=self.init_ap)
        self.model.beta = Var(self.model.f_s, within= PercentFraction, initialize=self.init_b)
        self.model.mu = Var(self.model.N_B, within = Reals, bounds = (-1.0,1.0),initialize=self.init_mu)
        self.model.C = Var(self.model.K, self.model.N_L, within = PercentFraction,initialize=self.init_C)
        self.model.P = Var(self.model.I,self.model.N_L,within = PercentFraction,initialize=self.init_P)
        # Variable to manage regularization L0
        self.model.z = Var(self.model.f_s, self.model.N_B, within=Reals, bounds = (-1.0,1.0),initialize=self.init_z)
       
        def Pr(model,i,tl):
            return  reduce(operator.mul,(1 / (1 + exp(-512*(   (sum(model.x[tb,i,j]*model.a[j,tb]for j in self.model.f_s)/len(self.model.f_s))-model.mu[tb]  )))  for tb in self.model.N_L_L[tl]),1)*reduce(operator.mul,(1-(1 / (1 + exp(-512*(   (sum(model.x[tr,i,j]*model.a[j,tr]for j in self.model.f_s)/len(self.model.f_s))-model.mu[tr]  )))) for tr in self.model.N_L_R[tl]),1) == model.P[i,tl]
        self.model.Pr = Constraint(self.model.I,self.model.N_L, rule=Pr)

        
        # We must add the following set of constraints for making a single class prediction at each leaf node:
        def class_in_leaf(model, tl):
            return  sum(model.C[k,tl] for k in self.model.K) == 1
        self.model.class_in_leaf = Constraint(self.model.N_L, rule=class_in_leaf)

        def leaf_in_class(model,k):
            return sum(model.C[k,tl] for tl in self.model.N_L) >=1
        self.model.leaf_in_class = Constraint(self.model.K, rule=leaf_in_class)
        
        #The following set of constraints uanbles to manage global regularization
        def global_min(model,f,tb):
            return model.beta[f]>= model.a_plus[f,tb]+model.a_minus[f,tb]
        self.model.globalmin = Constraint(self.model.f_s, self.model.N_B, rule=global_min)

        #The following set of constraints unables to manage local regularization
        def cuts_var(model,f,tb):
            return model.a_plus[f,tb]-model.a_minus[f,tb]==model.a[f,tb]
        self.model.cuts = Constraint(self.model.f_s,self.model.N_B, rule=cuts_var)
        
        # The following set of constraints are useful for L0 regularization
        def L0_par_mi(model,f,tb):
            return model.a[f,tb] <= model.z[f,tb]
        self.model.L0_min = Constraint(self.model.f_s, self.model.N_B, rule=L0_par_mi)
        
        def L0_par_ma(model,f,tb):
            return model.a[f,tb] >= - model.z[f,tb]
        self.model.L0_ma = Constraint(self.model.f_s, self.model.N_B, rule=L0_par_ma)
        return True
    
    # This method is an old version of methods to charge an objective function. Below i defined a gentle new version that deal with different models through a switcher method
    def objective(self,l_inf=0,l_l1=0,l_l2=0,l_l0=0,l_l0_inf=0,l_log=0, pr1=0, pr2=0):
        """ This method instantiates th Objective function to minimize.
        
        Params:
            l_inf (float) - regularization therm for L infinity norm
            l_l1  (float) - regularization therm for L 1 norm
            l_l2  (float) - regularization therm for L 2 norm
            l_l0  (float) - regularization therm for L 0 norm
            l_l0_inf(float) - regularization therm for L 0 inf norm
            l_log(float) - regularization therm for log approximation of L0 norm suggested in Rinaldi & Sciandrone
            pr1(float) - regularization therm for first attempt of approximation of L0 norm suggested in Rinaldi & Sciandrone
            pr2(float) - regularization therm for first attempt of approximation of L0 norm suggested in Rinaldi & Sciandrone
        
        Return:
            result (boolean) - objective function result
            
        """
        self.model.l_inf = l_inf
        self.model.l_l1 = l_l1
        self.model.l_l2 = l_l2
        self.model.l_l0 = l_l0
        self.model.l_l0_inf = l_l0_inf
        self.model.l_log = l_log 
        self.model.pr1 = pr1
        self.model.pr2 = pr2
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_inf*(sum(model.beta[j] for j in self.model.f_s))+self.model.l_l1*(sum(sum(model.a_plus[j,tb]+model.a_minus[j,tb] for tb in self.model.N_B)for j in self.model.f_s))+self.model.l_l2*(sum(sum(model.a[j,tb]**2 for tb in self.model.N_B) for j in self.model.f_s))+self.model.l_l0*sum(sum((1-exp(-model.alpha*model.z[j,tb])) for tb in self.model.N_B)for j in self.model.f_s)+self.model.l_l0_inf*sum((1-exp(-model.alpha*model.beta[j])) for j in self.model.f_s)+self.model.l_log*sum(sum( log(1e-6+model.z[j,tb]) for tb in self.model.N_B)for j in self.model.f_s)+self.model.pr1*sum(sum( 1e-6+model.z[j,tb] for tb in self.model.N_B)for j in self.model.f_s)+self.model.pr2*sum(sum( -1/(1e-6+model.z[j,tb]) for tb in self.model.N_B)for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function loaded")
        return True
    
    # In order to deal with switcher method i need to define several methods: one for each model we would like to test
    
    # Definition of objective function for L0 regularization approximation suggested in Mangasarian
    def l0(self):
        """ This method instantiates th Objective function to minimize for L0 regularization approximation.
        
        Return:
            result (boolean) - objective function result
            
        """

        self.model.l_l0 = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l0*sum(sum((1-exp(-model.alpha*model.z[j,tb])) for tb in self.model.N_B)for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l0 regularization is loaded")
        return True
    
    # Definition of objective function for L0 inf regularization approximation suggested in Rinaldi & Sciandrone
    def l0_inf(self):
        """ This method instantiates the Objective function to minimize for L0 inf regularization approximation.
        
        
        Return:
            result (boolean) - objective function result
            
        """

        self.model.l_l0_inf = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l0_inf*sum((1-exp(-model.alpha*model.beta[j])) for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l0_inf regularization is loaded")
        return True
    
    # Definition of objective function for L1 regularization
    def l1(self):
        """ This method instantiates the Objective function to minimize for L1 regularization.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        self.model.l_l1 = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l1*(sum(sum(model.a_plus[j,tb]+model.a_minus[j,tb] for tb in self.model.N_B)for j in self.model.f_s))
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l1 regularization is loaded")
        return True
    
    # Definition of objective function for L inf regularization
    def l_inf(self):
        """ This method instantiates th Objective function to minimize for L inf regularization.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        self.model.l_inf = self.reg_term
        
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_inf*(sum(model.beta[j] for j in self.model.f_s))
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l_inf regularization is loaded")
        return True
    
    # Definition of objective function for simple model
    def simple(self):
        """ This method instantiates the Objective function to minimize for simple model.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function loaded")
        return True
    
    # Definition of objective function for both regularization: L1 & L inf
    def both(self):
        """ This method instantiates the Objective function to minimize for both regularization: L1 & L inf.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        if len(self.reg_term)==1:
            self.model.l_inf = self.reg_term
            self.model.l_l1 = self.reg_term
        else:
            self.model.l_inf = self.reg_term[0]
            self.model.l_l1 = self.reg_term[1]
            
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_inf*(sum(model.beta[j] for j in self.model.f_s))+self.model.l_l1*(sum(sum(model.a_plus[j,tb]+model.a_minus[j,tb] for tb in self.model.N_B)for j in self.model.f_s))
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function loaded")
        return True
    
    # Definition of objective function for both approximation of l0 regularization: L0 & L0 inf
    def both_l0(self):
        """ This method instantiates the Objective function to minimize for both approximation of l0 regularization: L0 & L0 inf.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        # Since i have two regularization term i check how long the length of reg_term 
        if len(self.reg_term)==1:
            self.model.l_l0_inf = self.reg_term
            self.model.l_l0 = self.reg_term
        else:
            self.model.l_l0_inf = self.reg_term[0]
            self.model.l_l0 = self.reg_term[1]
            
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l0_inf*sum((1-exp(-model.alpha*model.beta[j])) for j in self.model.f_s)+self.model.l_l0*sum(sum((1-exp(-model.alpha*model.z[j,tb])) for tb in self.model.N_B)for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function loaded")
        return True
    
    def charge_of(self,argument,reg_term=0):
        """ This method instantiates the switcher in order to set the Objective function to minimize.
        
        Param:
            argument (string) - name of the model to charge i.e. "simple", "l0", "l1", "l_inf", "l0_inf", "both", "both_l0"
        
        Return:
            result (boolean) - objective function result
            
        """
        self.reg_term = reg_term
        switcher = {
            "simple": self.simple,
            "l1": self.l1,
            "l0": self.l0,
            "l_inf": self.l_inf,
            "l0_inf": self.l0_inf,
            "both": self.both,
            "both_l0": self.both_l0,
            }
        # Get the function from switcher dictionary
        func = switcher.get(argument, lambda: "You type in something wrong!")
        # Execute the function
        func()
        return True
    
    def solve(self):
        """This method invokes the Ipopt solver
        
        Return:
            result (boolean) - solve function result
        
        """
        solver = SolverFactory('ipopt',executable='C:/Users/antoc/Desktop/Ipopt-3.11.1-win64-intel13.1/bin/ipopt.exe')
        results = solver.solve(self.model) # ,tee=True show all the steps made by the solver
        return True
    
        # function to get init values
    def init_values(self):
        """ This method get initialization values of the variables defined in the model
        
        Return:
            result (array) - Initialization values
            
        """
        return [self.init_a,self.init_mu,self.init_C,self.init_P,self.init_p]
    
    # function to set init values
    def set_init(self,init):
        """ This method set initialization values of the variables defined in the model
        
        Param:
            init (array) - initialization values
            
        Return:
            result (boolean) - set_init function result
            
        """
        self.init_a = init[0]
        self.init_mu = init[1]
        self.init_C = init[2]
        self.init_P = init[3]
        if (len(init)>5):
            self.init_am = init[5]
            self.init_ap = init[6]
            self.init_b = init[7]
            self.init_z = init[8]
        print("Init values defined.")
        return True
    
    def set_lambdas(self,l_inf=0,l_l1=0,l_l2=0,l_l0=0,l0_inf=0,l_log = 0):
        """ This method set the regularization therms
        
        Params:
            l_inf (float) - regularization therm for L infinity norm
            l_l1  (float) - regularization therm for L 1 norm
            l_l2  (float) - regularization therm for L 2 norm
            l_log  (float) - regularization therm for L 0 norm log approximation
            l_l0  (float) - regularization therm for L 0 norm approximation
            l0_Inf  (float) - regularization therm for L 0 inf norm approximation
        Return:
            result (boolean) - set_lambdas function result
        
        """
        self.model.l_inf = l_global
        self.model.l_l1 = l_local
        self.model.l_l2 = l_l2
        self.model.l_l0 = l_l0
        self.model.l0_inf = l0_inf
        self.model.l_log = l_log
        return True
    
    # Function to store the variables results
    def extraction_va(self):
        """ This method stores the variables values at optimum
        
        Return:
            result (boolean) - extraction_va function result
        
        """
        mu = {str(self.model.mu[i]): self.model.mu[i].value for i in self.model.mu}
        a = {str(self.model.a[i]): self.model.a[i].value for i in self.model.a}
        C = {str(self.model.C[i]): self.model.C[i].value for i in self.model.C}
        am = {str(self.model.a_minus[i]): self.model.a_minus[i].value for i in self.model.a_minus}
        ap = {str(self.model.a_plus[i]): self.model.a_plus[i].value for i in self.model.a_plus}
        b = {str(self.model.beta[i]): self.model.beta[i].value for i in self.model.beta}
        P = {str(self.model.P[i]): self.model.P[i].value for i in self.model.P}
        self.var = {'mu': mu,'a':a ,'C':C,'am':am,'ap':ap,'b':b, 'P':P }
        
        return True
    
    #Calculate the predicted label of a single instance
    def comp_label(self,x):
        """ This method predictes label of a single instance
        
        Param:
            x (array) - single instance
        Return:
            result (integer) - label of class
        
        """
        # Prob is a function defined in maths_random: it calculates the probability of an individual falling into a given leaf node
        prob ={k : sum(Prob(self.model,self.index_features,self.var,x,i,self.initi,self.depth,self.percent)*self.var['C']['C['+str(k)+','+str(i)+']'] for i in self.model.N_L) for k in self.model.K}
        return int(max(prob, key=prob.get))
    
    #Generate a list of predicted labels for the test set
    def predicted_lab(self,X_test):
        """ This method predictes label of a several instances
        
        Param:
            X_test (pandas.dataframe) - test set of instances
        
        Return:
            label (array) - labels of class
        
        """
        label = []
        for i in range(0,len(X_test)):
            label.append(self.comp_label(X_test.iloc[i]))
        return label