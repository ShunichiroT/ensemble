# -*- coding: utf-8 -*-

import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats
import numpy as np

models = ['GAT','GBLUP','BayesB','RF','RKHS']
populations =  ['W22TIL01','W22TIL03','W22TIL11','W22TIL14','W22TIL25']
traits = ['DTA','KW','TILN','PLHT']

RESULT_PATH = "PATH"
SAVE_PATH = "PATH"

record = pd.DataFrame()   
result_prediction = pd.DataFrame()

cnt = 1
sample = 500

os.chdir(RESULT_PATH)
#GEBV matrix
data = pd.read_csv("result_teo_0.8_ver5_th0.95_concat_pred-result-all_0.8_test.csv").iloc[:,1:]

data_extracted = data[data['type'] == models[0]]
data_extracted = data_extracted.reset_index(drop=True)
label = data_extracted.iloc[:,1]
data_extracted = data_extracted.iloc[:,2:]

for i in range(len(models)):
   tmp = data[data['type'] == models[i]].iloc[:,0]
   tmp = tmp.reset_index(drop=True)
   data_extracted = pd.concat([data_extracted, tmp], axis=1)

data_extracted = pd.concat([data_extracted, label], axis=1)

for i in range(0,len(populations)):
    for j in range(0,len(traits)):
        if (i < 4 and j != 3) or (i == 4 and j < 4): 
            if traits[j]=="KW":
                sample+=500
                continue
            for k in range(sample-500,sample):
                for m in range(1):                                               
                           data_f = data_extracted[(data_extracted['pop'] == populations[i]) &
                                                   (data_extracted['ptype'] == traits[j]) &
                                                   (data_extracted['k'] == k+1) &
                                                   (data_extracted['repe'] == m)]
                           data_f = data_f.iloc[:,6:]
                           
                           data_f = data_f.reset_index(drop=True)
                           
                           ids = np.array(range(0,len(data_f.iloc[:,0])))
                           for q in range(1):
                               
                               for p in range(1):
                                   predicted = data_f.iloc[:,:-1].mean(axis=1)
                                   
                                   r = 0
                                   r2 = 0
                                   mse = 0
                        
                                   actual = data_f.iloc[:,-1].tolist()
                                   
                                   mse = mean_squared_error(actual, predicted)
                                   
                                   r = scipy.stats.pearsonr(actual, predicted)[0]
                                   
                                   r2 = r2_score(actual, predicted)
                                   
                                   print('MSE : ' + str(mse))
                                   print('Pearson r : ' + str(r))
                                   print('R2 : ' + str(r2))
                                   
                                   record = record.append({'Population': str(populations[i]),
                                                     'type': 'ensemble-average',
                                                     'Phenotype': str(traits[j]),
                                                     'sample': q+1,
                                                     'column':'Null',
                                                     'repeat': p+1,
                                                     'MSE':mse, 
                                                     'Pearson r':r,
                                                     'R2':r2
                                                     },ignore_index=True)
                                   
                                   os.chdir(SAVE_PATH)
                                   record.to_csv('assembled_avg_concat_0.8.csv')
                               
                cnt+= 1
            sample += 500