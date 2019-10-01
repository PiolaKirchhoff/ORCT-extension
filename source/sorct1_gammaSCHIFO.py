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
from itertools import compress
import pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from source.util1 import *
from source.maths1 import *


class SORCT_1:
    """ The class manage Sparsity in Optimal Randomized Classification Trees (SORCT) of Blanquero et Al. 2018
    and its extensions. The depth of the tree is set to 2.

    Parameters:
        dataset (pandas.dataframe)  - New data
        I_in_k (dictionary)         - Key: Class integer ; Values: array of instances indexes of that class 
        I_k (method)                - Method to manage the dictionary defined above to deal with Pyomo sets    
        init(array)                 - Initialization values
    
    """
    def __init__(self,dataset,data_test,I_in_k,I_k,init=[],d = {}):
        """ The constructor initializes dictionaries and methods to deal with decision tress topologies.
            There is a preprocessing phase to deal with dataset.By default the dataset is assumed as follows:
            - The column of the labels must be named "Classes"
            - The values that labels asummed are encoded by integers
        
        """
        self.B_in_NL = B_in_NL 
        self.B_in_NR = B_in_NR
        self.I_k = I_k
        self.my_W = cost(dataset)
        self.I_in_k = I_in_k
        self.my_x = my_train(dataset)
        
        self.BF_in_NL_R = BF_in_NL_R
        self.BF_in_NL_L = BF_in_NL_L
        self.number_f = len(dataset.columns)-1
        self.index_features = list(range(0,self.number_f))
        self.index_instances = list(dataset.index)
        self.dataset = dataset
        self.data_test = data_test
        self.init_a = init[0] if len(init) > 0 else np.random.uniform(low=-1.0, high=1.0, size=None)
        self.init_mu = init[1] if len(init) > 0 else np.random.uniform(low=-1.0, high=1.0, size=None)
        self.init_C = init[2] if len(init) > 0 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_P = init[3] if len(init) > 0 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_am = init[5] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_ap = init[6] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_b = init[7] if len(init) > 5 else np.random.uniform(low=0.0, high=1.0, size=None)
        self.init_z = init[8] if len(init) > 5 else np.random.uniform(low=-1.0, high=1.0, size=None)
        
        self.d = d
        
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
        self.model.W = Param(self.model.K, self.model.K, within=NonNegativeReals, initialize=self.my_W)
        self.model.x = Param(self.model.I, self.model.f_s, within=PercentFraction, initialize=self.my_x)
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
            return  reduce(operator.mul,(1 / (1 + exp(-512*(   (sum(model.x[i,j]*model.a[j,t]for j in self.model.f_s)/len(self.model.f_s))-model.mu[t]  )))  for t in self.model.N_L_L[tl]),1)*reduce(operator.mul,(1-(1 / (1 + exp(-512*(   (sum(model.x[i,j]*model.a[j,tr]for j in self.model.f_s)/len(self.model.f_s))-model.mu[tr]  )))) for tr in self.model.N_L_R[tl]),1) == model.P[i,tl]
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
    
    def objective(self,l_inf=0,l_l1=0,l_l2=0,l_l0=0,l_l0_inf=0,l_log=0, pr1=0, pr2=0):
        """ This method instantiates th Objective function to minimize.
        
        Params:
            l_inf (float) - regularization therm for L infinity norm
            l_l1  (float) - regularization therm for L 1 norm
            l_l2  (float) - regularization therm for L 2 norm
            l_l0  (float) - regularization therm for L 0 norm
        
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
    
    def l0(self):
        """ This method instantiates th Objective function to minimize.
        
        Return:
            result (boolean) - objective function result
            
        """

        self.model.l_l0 = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l0*sum(sum((1-exp(-model.alpha*model.z[j,tb])) for tb in self.model.N_B)for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l0 regularization is loaded")
        return True
    
    def l0_inf(self):
        """ This method instantiates th Objective function to minimize.
        
        
        Return:
            result (boolean) - objective function result
            
        """

        self.model.l_l0_inf = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l0_inf*sum((1-exp(-model.alpha*model.beta[j])) for j in self.model.f_s)
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l0_inf regularization isloaded")
        return True
    
    def l1(self):
        """ This method instantiates th Objective function to minimize.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        self.model.l_l1 = self.reg_term

        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_l1*(sum(sum(model.a_plus[j,tb]+model.a_minus[j,tb] for tb in self.model.N_B)for j in self.model.f_s))
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l1 regularization is loaded")
        return True
    
    def l_inf(self):
        """ This method instantiates th Objective function to minimize.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        self.model.l_inf = self.reg_term
        
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )+self.model.l_inf*(sum(model.beta[j] for j in self.model.f_s))
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function with l_inf regularization is loaded")
        return True
    
    def simple(self):
        """ This method instantiates th Objective function to minimize.
        
        
        Return:
            result (boolean) - objective function result
            
        """
        
        def cost_rule(model):
            return sum( sum( sum( model.P[i,t]* sum(model.W[k,j]*model.C[j,t] for j in self.model.K if k!=j)  for t in self.model.N_L) for i in self.model.I_k[k] ) for k in self.model.K )
        self.model.cost = Objective(rule=cost_rule, sense=minimize)
        print("Objective function loaded")
        return True
    
    def charge_of(self,argument,reg_term=0):
        
        self.reg_term = reg_term
        switcher = {
            "simple": self.simple,
            "l1": self.l1,
            "l0": self.l0,
            "l_inf": self.l_inf,
            "l0_inf": self.l0_inf,
            }
        # Get the function from switcher dictionary
        func = switcher.get(argument, lambda: "You type in something wrong!")
        # Execute the function
        func()
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
    
    def set_alpha(self,alp):
        self.model.alpha = alp
        return True
    
    def solve(self):
        """This method invokes the Ipopt solver
        
        Return:
            result (boolean) - solve function result
        
        """
        solver = SolverFactory('ipopt',executable='C:/Users/antoc/Desktop/Ipopt-3.11.1-win64-intel13.1/bin/ipopt.exe')
        results = solver.solve(self.model) # ,tee=True
        return True
    
    def value_obj(self):
        """ This method get Objective function value
        
        Return:
            result (float) - value_obj function result
        
        """
        return value((self.model.cost))
    
    def set_lambdas(self,l_inf=0,l_l1=0,l_l2=0,l_l0=0):
        """ This method set the regularization therms
        
        Params:
            l_inf (float) - regularization therm for L infinity norm
            l_l1  (float) - regularization therm for L 1 norm
            l_l2  (float) - regularization therm for L 2 norm
            l_l0  (float) - regularization therm for L 0 norm
            
        Return:
            result (boolean) - set_lambdas function result
        
        """
        self.model.l_inf = l_global
        self.model.l_l1 = l_local
        self.model.l_l2 = l_l2
        self.model.l_l0 = l_l0
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
        self.var = {'mu': mu,'a':a ,'C':C,'am':am,'ap':ap,'b':b }
        
        return True
    
    
    #Calculate the predicted label of a single instance
    def comp_label(self,x):
        """ This method predictes label of a single instance
        
        Param:
            x (array) - single instance
        Return:
            result (integer) - label of class
        
        """
        prob ={k : sum(Prob(self.model,self.index_features,self.var,x,i)*self.var['C']['C['+str(k)+','+str(i)+']'] for i in self.model.N_L) for k in self.model.K}
        return int(max(prob, key=prob.get))
    
    #Generate a list of predicted labels for the test set
    def predicted_lab(self,X_test):
        """ This method predictes label of a several instances
        
        Param:
            X_test (pandas.datafram) - test set of instances
        
        Return:
            label (array) - labels of class
        
        """
        label = []
        for i in range(0,len(X_test)):
            label.append(self.comp_label(list(X_test.iloc[i])))
        return label

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
    
    #Calculate the percentage of predictor variables per branch node (Local)
    def delta_l(self):
        """ This method calculate the percentage of predictor variables per branch node
        
        Return:
            result (float) - percentage of predictor variables per branch node
        
        """
        return round((sum(sum(1 if abs(self.var['a']['a['+str(j)+','+str(b)+']']) > 0.0000001 else 0 for j in self.model.f_s)/len(self.model.f_s) for b in self.model.N_B)/len(self.model.N_B))*100,4)

    #Calculate the percentage of predictor variables per tree (Global)
    def delta_g(self):
        """ This method calculate the percentage of predictor variables per tree
        
        Return:
            result (float) - percentage of predictor variables per tree
        
        """
        return round((sum(lit([1 if abs(self.var['a']['a['+str(j)+','+str(b)+']']) > 0.0000001 else 0 for b in self.model.N_B]) for j in self.model.f_s)/len(self.model.f_s))*100,4)
    
    def nbn_per_f(self,index):
        """ This method calculate the percentage of breanch nodes in which the feature of fixed index has an impact different from zero 
        
        Param:
            index (integer) - index of the feature
            
        Return:
            val (float) - percentage of predictor variables per tree
        
        """
        val = -1
        if index > self.number_f:
            print("The index is out of range.")
            return val
        else:
            inde = index - 1
            val = sum([1 if abs(self.var['a']['a['+str(inde)+','+str(b)+']']) > 0.0000001 else 0 for b in self.model.N_B])/3 
            return val
         
    def fs_selected(self,threshold = 1e-4):
        """ This method return a list of booleans to know which feature was selected during the training phase  
        
        Param:
            threshold (float) - threshold for the feature
            
        Return:
            lis       (list)  - list of boolean associated to predictor variables
        
        """
        lis = []
        for j in self.model.f_s:
            present = False
            for b in self.model.N_B:
                if (abs(self.var['a']['a['+str(j)+','+str(b)+']']) > threshold):
                    present = True
            lis.append(present)
        self.list_fs = lis
        return lis
    
    def df_results(self):
        """ This method return a list of booleans to know which feature was selected during the training phase  
        
        Param:
            threshold (float) - threshold for the feature
            
        Return:
            lis       (list)  - list of boolean associated to predictor variables
        
        """
        if self.list_fs:
            names_column = list(self.dataset.columns)
            n = list(compress(names_column, self.list_fs))
            return self.dataset.loc[:,n]
        else:
            return pd.DataFrame()
    
    def fs_for_kernel(self,threshold = 1e-4):
        """ This method return a list of booleans to know which feature was selected during the training phase  
        
        Param:
            threshold (float) - threshold for the feature
            
        Return:
            lis       (list)  - list of boolean associated to predictor variables
        
        """
        matrix_presence = {}
        for b in self.model.N_B:
            matrix_presence[b] = []
            for j in self.model.f_s:
                if (abs(self.var['a']['a['+str(j)+','+str(b)+']']) > threshold):
                    matrix_presence[b].append(j)
            
        self.dict_fs = matrix_presence
        
        return matrix_presence
    
    def update_rule(self,list_fs,num_node):
        
        lunghezza = int(self.number_f)
        # 1e-3 is and eps in order to manage empyt scenarios
        val =  [1e-3] * int(self.number_f*(self.number_f+1)/2)
        for l in list_fs:
            ix = 0
            agg = lunghezza
            for i in range (lunghezza):
                if i < l:
                    val[int(ix+l-i)] += abs(self.var['a']['a['+str(l)+','+str(num_node)+']'])/(len(list_fs)**2)
                    ix += agg
                    agg -= 1
                else:
                    val[int(ix+i-l)]+=abs(self.var['a']['a['+str(l)+','+str(num_node)+']'])/(len(list_fs)**2)
        return val
    
    def generate_prob(self,threshold = 1e-4):
        
        self.fs_for_kernel(threshold)
        
        matrix_prob = []
        for i in self.dict_fs:
            matrix_prob.append(self.update_rule(self.dict_fs[i],i))
            
        result = np.sum(matrix_prob,axis=0)
        res = result / sum(result)
        
        self.prob_couple = res
        
        return res
    
    def generate_df(self,threshold = 1e-4):
        """ This method return a list of booleans to know which feature was selected during the training phase  
        
        Param:
            threshold (float) - threshold for the feature
            
        Return:
            lis       (list)  - list of boolean associated to predictor variables
        
        """
        N = self.number_f
        final_l = self.fs_selected(threshold)
        my_prob = self.generate_prob(threshold)
        print(len(my_prob))
        couples = np.random.choice(range(0, int(self.number_f*(self.number_f+1)/2)), p=my_prob,size = N)
        list_couple = [False] * int(self.number_f*(self.number_f+1)/2)
        for j in couples:
            list_couple[j]= True
        new_fs = final_l+list_couple
        new_df = extended2_df(self.dataset)
        names_column = list(new_df.columns)
        n = list(compress(names_column, new_fs))
        
        return new_df.loc[:,n]
    
    def update_rule1(self,threshold = 1e-4):
        
        n = self.number_f
        final_l = [1 for j in self.model.f_s]
        #self.fs_selected(threshold)
        
        if not self.d:
            for j in range(n):
                if final_l[j]== True:
                    self.d[j] = [1 if i == j else 0 for i in range(n)]
                    
        my_prob = self.generate_prob(threshold)
        couples = np.random.choice(range(0, int(self.number_f*(self.number_f+1)/2)), p=my_prob,size = n)
        list_couple = [0] * int(self.number_f*(self.number_f+1)/2)
        
        for j in couples:
            list_couple[j]= 1
        
        w=[]
        for i in range(n):
            for j in range(n):
                if i<=j:
                    w = w + [[i,j]]
        
        for ix in range(len(list_couple)):
            if list_couple[ix] == 1:
                newcolumn = list(np.sum([ self.d[w[ix][0]],  self.d[w[ix][1]] ],axis=0))
        
        #check if it's good
                check = True
                for key in self.d:
                    if self.d[key] == newcolumn:
                        check = False
        
        #add to dictionary
                if check == True:
                    self.d[len(self.d)] = newcolumn
                else:
                    list_couple[ix] = 0
                    #  MANCA L'APPEND DELLA CLASSE??
        new_fs = final_l+list_couple
        new_df = extended2_df(self.dataset)
        
        new_test = extended2_df(self.data_test)
        names_column_test = list(new_test.columns)
        h = list(compress(names_column_test, new_fs))
        self.test_to_send = new_test.loc[:,h]
        
        print("INDICI:")
        print(new_fs)
        self.indeces_exdf = new_fs
        names_column = list(new_df.columns)
        n = list(compress(names_column, new_fs))
        
        return new_df.loc[:,n]
    
    def test_generation(self):
        
        
        return self.test_to_send
    
    def predicted(self):
        """ This method predictes label of a several instances
        
        Param:
            X_test (pandas.datafram) - test set of instances
        
        Return:
            label (array) - labels of class
        
        """
        label = []
        for i in range(0,len(self.data_test)):
            label.append(self.comp_label(list(self.data_test.iloc[i][:-1])))
        return label