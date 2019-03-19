# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:26:26 2019

@author: antoc
"""
import numpy as np
import pandas as pd
import math
from functools import reduce
import operator
from pyomo.environ import *
from pyomo.opt import SolverFactory
from source.util import *
from source.maths import *


class ORCT_2:

    def __init__(self,dataset,I_in_k,I_k,init_a,init_mu,init_C,init_P,init_p):
        self.B_in_NL = B_in_NL 
        self.B_in_NR = B_in_NR
        self.I_k = I_k
        self.my_W = cost(dataset)
        self.I_in_k = I_in_k
        self.my_x = my_train(dataset)
        
        self.BF_in_NL_R = BF_in_NL_R
        self.BF_in_NL_L = BF_in_NL_L
        self.index_features = list(range(0,len(dataset.columns)-1))
        self.index_instances = list(dataset.index)
        
        self.init_a = init_a
        self.init_mu = init_mu
        self.init_C = init_C
        self.init_P = init_P
        self.init_p = init_p
        
        self.createModel()
        
    def createModel(self):
        self.model = ConcreteModel() #ConcretModel()
        self.model.I = Set(initialize=set(i for k in self.I_in_k for i in self.I_in_k[k]))
        self.model.K = Set(initialize=self.I_in_k.keys())
        self.model.I_k = Set(self.model.K,initialize=self.I_k)    
        self.model.f_s =Set(initialize=self.index_features)
        self.model.N_B = Set(initialize=set(i for k in self.BF_in_NL_R for i in self.BF_in_NL_R[k]))
        self.model.N_L = Set(initialize=self.BF_in_NL_R.keys())
        self.model.N_L_R = Set(self.model.N_L,initialize=self.B_in_NR)
        self.model.N_L_L = Set(self.model.N_L,initialize=self.B_in_NL)
        self.model.W = Param(self.model.K, self.model.K, within=NonNegativeReals, initialize=self.my_W)
        self.model.x = Param(self.model.I, self.model.f_s, within=PercentFraction, initialize=self.my_x)
        self.model.a = Var(self.model.f_s, self.model.N_B, within=Reals, bounds = (-1.0,1.0),initialize=self.init_a)
        self.model.mu = Var(self.model.N_B, within = Reals, bounds = (-1.0,1.0),initialize=self.init_mu)
        self.model.C = Var(self.model.K, self.model.N_L, within = PercentFraction,initialize=self.init_C)
        self.model.P = Var(self.model.I,self.model.N_L,within = PercentFraction,initialize=self.init_P)
        self.model.p = Var(self.model.I,self.model.N_B,within = PercentFraction,initialize=self.init_p)
        
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        def Pr(model,i,tl):
            return  reduce(operator.mul,(model.p[i,t] for t in self.model.N_L_L[tl]),1)*reduce(operator.mul,(1-model.p[i,tr] for tr in self.model.N_L_R[tl]),1) == model.P[i,tl]
        self.model.Pr = Constraint(self.model.I,self.model.N_L, rule=Pr)

        def pr(model, i , tb):
            return 1 / (1 + exp(-512*(   (sum(model.x[i,j]*model.a[j,tb]for j in self.model.f_s)/4)-model.mu[tb]  ))) ==model.p[i,tb]
        self.model.pr = Constraint(self.model.I,self.model.N_B, rule=pr)
        
        # We must add the following set of constraints for making a single class prediction at each leaf node:
        def class_in_leaf(model, tl):
            return  sum(model.C[k,tl] for k in self.model.K) == 1
        self.model.class_in_leaf = Constraint(self.model.N_L, rule=class_in_leaf)

        def leaf_in_class(model,k):
            return sum(model.C[k,tl] for tl in self.model.N_L) >=1
        self.model.leaf_in_class = Constraint(self.model.K, rule=leaf_in_class)
        
    def init_values(self):
        return [self.init_a,self.init_mu,self.init_C,self.init_P,self.init_p]
    
    def solve(self):
        """Solve the model."""
        solver = SolverFactory('ipopt',executable='C:/Users/antoc/Desktop/Ipopt-3.11.1-win64-intel13.1/bin/ipopt.exe')
        results = solver.solve(self.model)
        
    def value_obj(self):
        return value((self.model.cost))
    