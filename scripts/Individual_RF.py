# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats
import os
import numpy as np

REPEAT = 1
SAMPLE = 5
NUM = 1
cnt = 0 
cnt_sample = 500

pat = [["W22TIL01",0,5,[1,2],"DTA"],
        ["W22TIL01",0,5,[5,6],"KW"],
        ["W22TIL01",0,5,[9,10],"TILN"],
        ["W22TIL01",0,5,[14],"PLHT"],
        ["W22TIL03",0,5,[1,2],"DTA"],
        ["W22TIL03",0,5,[5,6],"KW"],
        ["W22TIL03",0,5,[9,10],"TILN"],
        ["W22TIL11",0,5,[1,2],"DTA"],
        ["W22TIL11",0,5,[5,6],"KW"],
        ["W22TIL11",0,5,[9,10],"TILN"],
        ["W22TIL14",0,5,[2,3],"DTA"],
        ["W22TIL14",0,5,[6,7],"KW"],
        ["W22TIL14",0,5,[10,11],"TILN"],
        ["W22TIL14",0,5,[15],"PLHT"],
        ["W22TIL25",0,5,[3,4],"DTA"],
        ["W22TIL25",0,5,[7,8],"KW"],
        ["W22TIL25",0,5,[11,12],"TILN"],
        ["W22TIL25",0,5,[15,16],"PLHT"],
        ]

record = pd.DataFrame()
f_imp = pd.DataFrame() 
result_prediction = pd.DataFrame()
result_prediction_test = pd.DataFrame()
 
GENO_PATH = "PATH"
PHENO_PATH = "PATH"
SAVE_PATH = "PATH"
IMP_PATH = "PATH"

os.chdir(IMP_PATH)
samples = pd.read_csv("samples_0.8.csv").iloc[:,1:]

for i in range(len(samples.iloc[:,0])):
    for j in range(len(samples.iloc[0,:])):
         samples.iloc[i,j] = [i for i in map(int, samples.iloc[i,j][1:-1].split(','))]

result_prediction = pd.DataFrame()

for q in range(0,len(pat)):
    if q == 3 or q == 13:
        cnt += 1
        continue
    else:
        if pat[q][4] =='KW':
            cnt_sample += 500
            cnt += 2
            continue
        for n in range(0,len(pat[q][3]),2):

            os.chdir(GENO_PATH)
            data1 = pd.read_csv(pat[q][0] + "_pruned_0.8_imputed.csv").iloc[:, 1:]
            data2 = pd.read_csv(pat[q][0] + "_pruned_0.8_imputed.csv").iloc[:, 1:]
            data = pd.concat([data1,data2])

            os.chdir(PHENO_PATH)
            data_pheno1 = pd.read_csv(pat[q][0] + "_pheno.csv").iloc[:, pat[q][3][n]]
            data_pheno2 = pd.read_csv(pat[q][0] + "_pheno.csv").iloc[:, pat[q][3][n+1]]
            data_pheno = pd.concat([data_pheno1, data_pheno2])

            data['target'] = data_pheno
            data = data.dropna(subset=['target'])
            data = data.dropna(axis=1,how='any')

            data1['target'] = data_pheno1
            data1 = data1.dropna(subset=['target'])
            data1 = data1.dropna(axis=1,how='any')
            
            data2['target'] = data_pheno2
            data2 = data2.dropna(subset=['target'])
            data2 = data2.dropna(axis=1,how='any')
            
            year = [[0]] * len(data1)
            year = year + [[1]] * len(data2)
            
            data = data.reset_index(drop=True)
            data.insert(0,'year',pd.DataFrame(year),False)

            ids = np.array(range(0,len(data.iloc[:,0])))
            
            for sample in range(cnt_sample-500,cnt_sample):
                
                data_train=data[data.index.isin(samples.iloc[sample,0])]
                data_test=data[data.index.isin(samples.iloc[sample,1])]
        
                train_x, train_y = data_train.iloc[:,:-1], data_train.iloc[:,-1]
                test_x, test_y = data_test.iloc[:,:-1], data_test.iloc[:,-1]
     
                while True:
                    if NUM > REPEAT:
                        NUM = 1
                        break
        
                    rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)

                    rf.fit(train_x, train_y)
                   
                    predicted = rf.predict(test_x)
                    
                    r = 0
                    r2 = 0
                    mse = 0
        
                    actual = test_y.values.tolist()
                    
                    mse = mean_squared_error(actual, predicted)
                    
                    r = scipy.stats.pearsonr(actual, predicted)[0]
        
                    r2 = r2_score(actual, predicted)
                    
                    result_pred = pd.concat([pd.DataFrame(predicted),
                                             pd.DataFrame(actual)],axis=1)
                    result_pred.columns = ['predicted','actual']
                    result_pred['pop'] = pat[q][0]
                    result_pred['sample'] = sample+1
                    result_pred['id'] = 'NULL'
                    result_pred['phenotype'] = pat[q][4]
                    result_pred['repeat'] = NUM
                    
                    result_prediction_test = pd.concat([result_prediction_test, result_pred],axis=0)
                    
                    predicted = rf.predict(train_x)
                    actual = train_y.values.tolist()
                    
                    result_pred = pd.concat([pd.DataFrame(predicted),
                                             pd.DataFrame(actual)],axis=1)
                    result_pred.columns = ['predicted','actual']
                    result_pred['pop'] = pat[q][0]
                    result_pred['sample'] = sample+1
                    result_pred['id'] = 'NULL'
                    result_pred['phenotype'] = pat[q][4]
                    result_pred['repeat'] = NUM
                    
                    result_prediction = pd.concat([result_prediction,result_pred],axis=0)
                    
                    print('Population : ' + str(pat[q][0]))
                    print('Phenotype : ' + str(pat[q][4]))
                    print('column : ' + str(pat[q][3][n]))
                    print('MSE : ' + str(mse))
                    print('Pearson r : ' + str(r))
                    print('R2 : ' + str(r2))
        
                    tmp = pd.DataFrame([{'Population': str(pat[q][0]),
                                          'Phenotype': pat[q][4],
                                          'column':pat[q][3][n],
                                           'repeat': NUM,
                                           'sample': sample+1,
                                           'Pearson r':r,
                                           'MSE':mse, 
                                           'R2':r2,
                                          }]) 
                     
                    record = pd.concat([record,tmp],ignore_index=True)
        
                    f_imp = pd.concat([f_imp,pd.DataFrame(rf.feature_importances_, index=rf.feature_names_in_).T], ignore_index = True)

                    os.chdir(SAVE_PATH)
                    record.to_csv('RF_0.8.csv')
                    f_imp.to_csv('RF_0.8_imp.csv')
                    result_prediction.to_csv('RF_0.8_result.csv')
                    result_prediction_test.to_csv('RF_0.8_result_test_concat.csv')
                    NUM += 1
            
            cnt_sample += 500     