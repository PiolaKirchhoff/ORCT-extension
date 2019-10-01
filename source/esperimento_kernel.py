# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:40:18 2019

@author: antoc
"""

# dataframe management
import pandas as pd
import math
import numpy as np
import random
import sklearn
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from pyomo.environ import *
from pyomo.opt import SolverFactory
from source.util import *
from source.maths1 import *
from source.sorct import *
from sklearn.model_selection import KFold
from source.kernel_extended import *


# In order to deal with scalability you can use command line as follow:
# argv[1]: name of the csv file i.e. IrisCategorical,Seeds,etc...
# argv[2]: number of iterations for kernel expansion

####        LISTA DI TERMINI DI REGOLARIZZAZIONE ordine 'simple' 'l1' 'l_inf' 'l0' 'l0_inf' 'both' 'both_l0'
list_reg = []

csv_name = sys.argv[1]
csv_name +='.csv'
iterations = sys.argv[2]


if len(sys.argv) == 3:
   try:

       print("Experiments on csv '%s' will start soon."%(csv_name))
       
   except IOError:
       print(sys.stderr, 'Error opening %s\n' %sys.argv[1])
       sys.exit(1)
else:
   f = sys.stdin
   
if csv_name == 'IrisCategorical.csv' or csv_name == 'wine_dataset.csv':
    data = pd.read_csv(csv_name)
else:
    data = pd.read_csv(csv_name, sep = ';')

data_st = data.copy()
data_st = data_st.replace('?', float('nan'))
data_st = data_st.dropna()

if csv_name == 'car.csv':
    df = pd.get_dummies(data_st.iloc[:,:-1])
    data_st = pd.concat([df, data_st.iloc[:,-1]], axis=1, join_axes=[df.index])    

if csv_name == 'IrisCategorical.csv' or csv_name == 'wisconsin.csv':
    data = data.drop('Id', axis=1)


if extended:
    data_st = extended2_df(data_st)

# Rename the column with the label with generic name 'Classes'
data.rename(columns={'Species':'Classes'}, inplace=True)
data_st.rename(columns={'Species':'Classes'},inplace=True)

#The name of the classes K
classes = data_st['Classes'].unique().tolist()
classes_en = [i for i in range(len(classes))] 
if csv_name !='car.csv':
    
    # Setting of the Scaling (to interval (0,1)) phase
    scaler = MinMaxScaler()

    #Preprocessing: we get the columns names of features which have to be standardized
    columns_names = list(data_st)
    index_features = list(range(0,len(data_st.columns)-1))

    
    #Scaling phase
    data_st[columns_names[0:len(index_features)]] = scaler.fit_transform(data_st[columns_names[0:len(index_features)]])

#Encoder processing
le = preprocessing.LabelEncoder()
le.fit(data_st['Classes'])

data_st['Classes'] = le.transform(data_st['Classes']) 


data_std = data_st.sample(frac= 1 ,random_state = 23).reset_index(drop=True)
data_ex =extended2_df(data_std)
data_std = data_std.replace(1.0000000000000002, 1)
data_std = data_std.replace(1.0000000000000004, 1)
data_std = data_std.replace(1.0000000000000009, 1)
data_ex = data_ex.replace(1.0000000000000002, 1)
data_ex = data_ex.replace(1.0000000000000004, 1)
data_ex = data_ex.replace(1.0000000000000009, 1)



init = {0:{'init_data': [1324, 4162, 2354, 8395, 2009, 770, 166, 281, 6181, 901, 405],
 'init_var': [[8838, 8390, 6503, 1145, 2352, 5128, 5022, 8365, 427],
  [2569, 8549, 9216, 8087, 5029, 9208, 2576, 6217, 75],
  [517, 230, 246, 980, 8429, 59, 617, 8361, 2983],
  [4895, 2471, 3107, 3044, 1163, 1173, 9186, 2324, 1278],
  [8236, 2191, 9127, 3335, 5876, 7880, 6096, 3163, 649],
  [2784, 5850, 2161, 512, 758, 1737, 829, 3741, 943],
  [7951, 8064, 6754, 2009, 6507, 8895, 8320, 7779, 792],
  [4138, 5831, 5883, 9000, 6780, 9391, 8144, 9610, 109],
  [486, 21, 776, 958, 557, 494, 319, 12, 3452],
  [7863, 2522, 6532, 7737, 7316, 5823, 3838, 2940, 9528]]},1:{'init_data': [7843, 8131, 4934, 161, 9631, 5578, 4931, 231, 7536, 8207],
 'init_var': [[5668, 5281, 7793, 4429, 2783, 7606, 4341, 5894, 6543],
  [975, 7060, 4741, 9209, 2105, 3646, 6741, 8892, 2495],
  [9700, 8534, 1261, 6902, 4908, 2515, 367, 7819, 482],
  [3943, 9884, 4456, 4375, 7972, 2599, 9291, 3532, 8572],
  [7721, 9316, 3870, 4774, 7125, 3187, 9795, 7859, 6606],
  [6633, 6117, 3147, 7950, 2069, 5572, 9357, 2255, 122],
  [2679, 2932, 4008, 5512, 7700, 4802, 3509, 5374, 7389],
  [7861, 2602, 5281, 588, 4387, 6173, 301, 4661, 5673],
  [9876, 3657, 6329, 598, 3356, 4538, 9901, 7345, 8386],
  [6145, 1397, 6048, 6418, 1022, 9434, 9519, 5398, 3713]]},2:{'init_data': [9758, 2137, 193, 4914, 162, 5362, 3807, 3366, 2712, 5171],
 'init_var': [[8123, 5976, 3520, 6528, 7773, 580, 705, 6538, 3107],
  [5564, 4512, 5405, 120, 9698, 5741, 5906, 6063, 954],
  [8207, 6339, 2757, 9767, 8697, 6560, 8190, 5056, 5075],
  [539, 3446, 3120, 6229, 4172, 8295, 4263, 299, 3713],
  [5401, 5610, 2267, 9183, 2762, 3842, 3351, 8946, 4199],
  [5507, 1001, 1384, 1820, 8431, 5453, 6159, 7836, 843],
  [8049, 8733, 8615, 3346, 2368, 598, 4681, 8932, 4177],
  [1380, 2669, 1797, 5920, 5491, 4652, 8194, 8706, 1263],
  [6638, 1221, 2353, 9760, 1658, 6128, 2308, 1228, 4870],
  [1197, 8588, 6200, 8402, 7456, 1726, 2629, 2743, 3388]]},3:{'init_data': [8239, 792, 1840, 8413, 4984, 4575, 8507, 5829, 4738, 8281],
 'init_var': [[1193, 4012, 2358, 4066, 874, 7317, 6400, 7311, 2436],
  [8406, 6189, 2800, 7504, 998, 9293, 3898, 4892, 5759],
  [1200, 3607, 181, 6763, 4200, 2160, 900, 3080, 971],
  [6252, 9731, 1005, 2973, 6243, 5990, 1988, 8, 2671],
  [4957, 4445, 1988, 4350, 378, 5012, 2794, 1941, 9061],
  [2847, 3491, 9797, 815, 5582, 332, 9676, 8121, 8314],
  [1571, 6399, 9128, 7376, 8030, 7948, 1710, 1, 4498],
  [4951, 9040, 8999, 3242, 6429, 4690, 4387, 8471, 7979],
  [1304, 1940, 4550, 9277, 9441, 8721, 1367, 2934, 5676],
  [8548, 7960, 8439, 9718, 29, 3397, 9782, 5, 6823]]},4:{'init_data': [7038, 6412, 6665, 5020, 5642, 9471, 8653, 576, 2994, 8643],
 'init_var': [[9306, 7994, 4492, 9935, 6536, 9779, 5858, 1499, 7541],
  [5404, 1143, 1248, 9481, 723, 8208, 3948, 4273, 929],
  [6300, 4871, 4917, 6038, 2737, 5862, 5408, 5011, 1209],
  [2405, 1284, 2256, 6323, 9474, 5782, 2455, 1190, 5528],
  [2575, 9377, 7702, 5043, 11, 4686, 4077, 554, 9490],
  [5838, 1325, 7791, 702, 3796, 3207, 1613, 3922, 6429],
  [9027, 3724, 7750, 4840, 2821, 7723, 4086, 6424, 4309],
  [2703, 887, 655, 850, 2698, 6244, 7428, 9854, 6272],
  [407, 1344, 3585, 133, 1398, 9091, 6380, 6763, 7605],
  [6052, 652, 4162, 5313, 2705, 104, 2059, 7095, 3801]]}}


fin_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
dl_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
dg_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
t_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}

fin = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
fin_dl = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
fin_dg = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
fin_t = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}

for i in fin:
        for j in range(iterations):
            fin[i][j] = []
            fin_dl[i][j] = []
            fin_dg[i][j] = []
            fin_t[i][j] = []
            
for p in range(5):
    initialization = init[p]
    
    acc_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
    delta_l_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
    delta_g_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
    time_ex = {'simple':[], 'l1':[], 'l_inf':[], 'l0':[],'l0_inf':[],'both':[],'both_l0':[]}
    
    acc = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
    delta_l = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
    delta_g = {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
    tim =  {'simple':{}, 'l1':{}, 'l_inf':{}, 'l0':{},'l0_inf':{},'both':{},'both_l0':{}}
    for i in acc:
        for j in range(iterations):
            acc[i][j] = []
    
    lambdas = 1/(3*(len(data.columns)-1))
    l = 1/(len(data.columns)-1)
    
    k_fold = KFold(n_splits=5)
    
    k = 0

    for train_indices, test_indices in k_fold.split(data_std):
    
        df_train = data_std.iloc[train_indices].reset_index(drop=True)
        df_train_ex2 = data_ex.iloc[train_indices].reset_index(drop=True)
    
        df_test = data_std.iloc[test_indices].reset_index(drop=True)
    
        X_test = data_std.iloc[test_indices,0:data_std.shape[1]-1]
        X_test_ex = data_ex.iloc[test_indices,0:data_ex.shape[1]-1]
    
        y_test = data_std.iloc[test_indices,data_std.shape[1]-1:data_std.shape[1]]
        I_in_k = inst_class(df_train_ex2)
        
        if len(classes)==3:
                def I_k(model,i):
                    if i==0:
                        return I_in_k[0]
                    elif i==1:
                        return I_in_k[1]
                    elif i==2:
                        return I_in_k[2]
            elif len(classes)==2:
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
                    elif i ==3:
                        return I_in_k[3]
        
        # INITIALIZATION FOR VARIABLES
        np.random.seed(initialization['init_var'][k][0])
        init_a = np.random.uniform(low=-1.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][1])
        init_mu = np.random.uniform(low=-1.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][2])
        init_C = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][3])
        init_P = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][4])
        init_p = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][5])
        init_beta = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][6])
        init_am = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][7])
        init_ap = np.random.uniform(low=0.0, high=1.0, size=None)
        np.random.seed(initialization['init_var'][k][8])
        init_z = np.random.uniform(low=0.0, high=1.0, size=None)
        k += 1
        
        ini = [init_a,init_mu,init_C,init_P,init_p,init_am,init_ap,init_beta,init_z]
    
        for i in acc_ex:
            model = SORCT(df_train_ex2,I_in_k,I_k)
            model.set_init(ini)
            model.createModel()
            model.set_alpha(5)
            
            # 'l1' 'l_inf' 'l0' 'l0_inf' 'both' 'both_l0'
            if i=='l1':
                lam = list_reg[0]
            elif i=='l_inf':
                lam = list_reg[1]
            elif i=='l0':
                lam = list_reg[2]
            elif i=='l0_inf':
                lam = list_reg[3]
            elif i=='both':
                lam = list_reg[4]
            elif i== 'both_l0':
                lam = list_reg[5]
            else:
                lam = 0
            model.charge_of(i,lam)
            
            start = time.time()
            try:
                model.solve()
            except ValueError:
                print ('Invalid value!')
            try:
                model.solve()
            except ValueError:
                print ('Invalid value!')
            end = time.time()
            
            model.extraction_va()
            p_s = model.predicted_lab(X_test_ex)
            acc_ex[i].append(model.accuracy(p_s,y_test.values))    
            delta_l_ex[i].append(model.delta_l())
            delta_g_ex[i].append(model.delta_g())
            time_ex[i].append(end-start)
            
            prova = extended_kernel(df_train,df_test,N=iterations,type_reg=i, reg_term =lam, ini=ini)
        
            for j in range(iterations):
                print("It: %d"%j)
                prova.update()
                p = prova.predicted()
                acc[i][j].append(prova.accuracy(p,y_test.values))
                delta_l[i][j].append(prova.learners[-1].delta_l())
                delta_g[i][j].append(prova.learners[-1].delta_g())
                tim[i][j].append(prova.time)
                
    for i in acc_ex:
        fin_ex[i].append(np.mean(acc_ex[i]))
        dl_ex[i].append(np.mean(delta_l_ex[i]))
        dg_ex[i].append(np.mean(delta_g_ex[i]))
        t_ex[i].append(np.mean(time_ex[i])) 
    for i in acc:
        for j in acc[i]:
            fin[i][j].append(np.mean(acc[i][j]))
            fin_dl[i][j].append(np.mean(delta_l[i][j]))
            fin_dg[i][j].append(np.mean(delta_g[i][j]))
            fin_t[i][j].append(np.mean(tim[i][j]))
  
print("EXTENDED")
print("Accuracy:")          
for i in acc_ex:
    print(i+str(":"))
    print(np.mean(fin_ex[i]))
print("Delta local:")          
for i in acc_ex:
    print(i+str(":"))
    print(np.mean(dl_ex[i]))
print("Delta global:")          
for i in acc_ex:
    print(i+str(":"))
    print(np.mean(dg_ex[i]))
print("Time:")          
for i in acc_ex:
    print(i+str(":"))
    print(np.mean(t_ex[i]))
    
print("Kernel Expansion")
print("Accuracy:")
for i in fin:
    print(i+str(":"))
    for j in fin[i]:
        print("It:"+str(j))
        print(np.mean(fin[i][j]))
print("Delta local:")
for i in fin:
    print(i+str(":"))
    for j in fin[i]:
        print("It:"+str(j))
        print(np.mean(fin_dl[i][j]))
print("Delta global:")
for i in fin:
    print(i+str(":"))
    for j in fin[i]:
        print("It:"+str(j))
        print(np.mean(fin_dg[i][j]))
print("Time:")
for i in fin:
    print(i+str(":"))
    for j in fin[i]:
        print("It:"+str(j))
        print(np.mean(fin_t[i][j]))