# -*- coding: utf-8 -*-
#import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, Linear, to_hetero_with_bases
from torch_geometric.loader import HGTLoader
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats
import math
import os

REPEAT = 1
num = 0
HEADS = 8
cnt = 0
cnt_sample = 500

patterns = [["W22TIL01",0,5,[1,2],"DTA"],
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

for z in range(0,len(patterns)):
    if z == 3 or z == 13:
        cnt += 1
        continue
    else:
        pat = patterns[z]
        if pat[4] =='KW':
            cnt_sample += 500
            cnt += 2
            continue
        for yr in range(0,len(pat[3]),2):
            os.chdir(GENO_PATH)
            data1 = pd.read_csv(pat[0] + "_pruned_0.8_imputed.csv").iloc[:, 1:]
            data2 = pd.read_csv(pat[0] + "_pruned_0.8_imputed.csv").iloc[:, 1:]
            data = pd.concat([data1,data2])

            os.chdir(PHENO_PATH)
            data_pheno1 = pd.read_csv(pat[0] + "_pheno.csv").iloc[:, pat[3][yr]]
            data_pheno2 = pd.read_csv(pat[0] + "_pheno.csv").iloc[:, pat[3][yr+1]]
            data_pheno = pd.concat([data_pheno1, data_pheno2])
            
            data['target'] = data_pheno
            data = data.dropna(subset=['target'])
            data = data.dropna(axis=1,how='any')
            
            data_QTL = data.iloc[:,:-1]
            data_pheno = data.iloc[:,-1]
            
            data_pheno = data_pheno.reset_index(drop=True)
            
            data1['target'] = data_pheno1
            data1 = data1.dropna(subset=['target'])
            data1 = data1.dropna(axis=1,how='any')
            
            data2['target'] = data_pheno2
            data2 = data2.dropna(subset=['target'])
            data2 = data2.dropna(axis=1,how='any')
            
            os.chdir(IMP_PATH)
            data_rf_imp = pd.read_csv("RF_SNPs_importance_avg_0.8_imputed.csv").iloc[:,1:]
            
            delete = []
            data_rf_imp_target = data_rf_imp.iloc[[cnt,cnt+1],:].mean()
            for i in range(len(data_rf_imp_target)):
                if data_rf_imp_target.index[i] not in data_QTL.columns:
                    delete.append(i)
            data_rf_imp_target = data_rf_imp_target.drop(data_rf_imp_target.index[delete])
            data_rf_imp_target = data_rf_imp_target[data_rf_imp_target >= data_rf_imp_target.quantile(0.75)].index.values
          
            data_columns = data_QTL.columns.values
            delete = []
            for i in range(len(data_QTL.iloc[0,:])):
                if data_columns[i] not in data_rf_imp_target:
                    delete.append(i)
            
            data_QTL = data_QTL.drop(data_QTL.columns[delete],axis=1)
            cnt += 2
            
            x_qtl = []
            for i in range(len(data_QTL.iloc[0,:])):
                x_qtl.append(list(map(lambda x: [x], data_QTL.iloc[:,i].to_list())))
            x_qtl = np.array(x_qtl)

            x_pheno = [[0]] * len(data1)
            x_pheno = x_pheno + [[1]] * len(data2)
                
            y_pheno = np.array(list(map(lambda x: [x], data_pheno.to_list())), dtype='float32')

            edges = np.array(range(0,len(data_QTL.iloc[:,0])))
            
            ids = np.array(range(0,len(data_QTL.iloc[:,0])))
            
            data = HeteroData()
            
            data['pheno'].x = torch.tensor(x_pheno, dtype=torch.float)
            data['pheno'].y = torch.from_numpy(y_pheno)
            
            for i in range(len(x_qtl)):
                data['qtl_'+str(i+1)].x = torch.tensor(x_qtl[i], dtype=torch.float)  
                data['qtl_'+str(i+1),'affect','pheno'].edge_index = torch.stack([torch.from_numpy(edges).to(torch.long),torch.from_numpy(edges).to(torch.long)], dim=0)
             
            for sample in range(cnt_sample-500,cnt_sample):
                
                train_id = samples.iloc[sample,0] 
                test_id = samples.iloc[sample,1]
                
                mask_train = np.array([])
                mask_valid = np.array([])
                mask_test = np.array([])
                for i in range(len(data_QTL.iloc[:,0])):
                    if i in train_id:
                        mask_train = np.append(mask_train, True)
                        mask_valid = np.append(mask_valid, False)
                        mask_test = np.append(mask_test,False)   
                    else:
                        mask_train = np.append(mask_train, False)
                        mask_valid = np.append(mask_valid, False)
                        mask_test = np.append(mask_test,True) 
                
                data['pheno'].train_mask = torch.from_numpy(mask_train).to(torch.bool)
                data['pheno'].val_mask = torch.from_numpy(mask_valid).to(torch.bool)
                data['pheno'].test_mask = torch.from_numpy(mask_test).to(torch.bool)

                
                activate_functions = [3]
                regularisation = [5e-4] 
                learning_rate = [0.005]
                dropout = [0.9]
                epoch = [150]
                channels = [20]
                
                comb = []
                valid_loss = np.inf
                chosen_model = 'init'
                best_index = 0

                for i in range(len(activate_functions)):
                    for j in range(len(regularisation)):
                        for k in range(len(learning_rate)):
                            for l in range(len(dropout)):
                                for m in range(len(epoch)):
                                    for n in range(len(channels)):
                                        t = []
                                        t.append(activate_functions[i])
                                        t.append(regularisation[j])
                                        t.append(learning_rate[k])
                                        t.append(dropout[l])
                                        t.append(epoch[m])
                                        t.append(channels[n])
                                        comb.append(t)
                while True:
                    if num >= REPEAT:
                        num = 0
                        break

                    for q in range(len(comb)):
                        class GAT(torch.nn.Module):
                            def __init__(self, hidden_channels, out_channels, dpout):
                                super().__init__()
                                self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=HEADS, concat=True, dropout=dpout)
                                self.lin1 = Linear(-1, hidden_channels*HEADS)
                                self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=HEADS, concat=True, dropout=dpout)
                                self.lin2 = Linear(-1, hidden_channels*HEADS)
                                self.conv3 = GATv2Conv((-1,-1), out_channels, add_self_loops=False, heads=HEADS, concat=False, dropout=dpout)
                                self.lin3 = Linear(-1, out_channels)
                             
                            def forward(self, x, edge_index):
                                x = self.conv1(x, edge_index) + self.lin1(x)
                                if comb[q][0] == 1:
                                    x = F.leaky_relu(x)
                                elif comb[q][0] == 2:
                                    x = F.relu(x)
                                elif comb[q][0] == 3:
                                    x = F.elu(x)
                                else:
                                    x = F.tanh(x)
                                x = self.conv2(x, edge_index) + self.lin2(x)
                                x = F.elu(x)
                                x = self.conv3(x, edge_index) + self.lin3(x)
                                return x

                        model = GAT(hidden_channels=comb[q][5], out_channels=1, dpout=comb[q][3])
                        model = to_hetero_with_bases(model, data.metadata(), num_bases=5)
                       
                        train_loader = HGTLoader(data, 
                                                num_samples=[100],
                                                shuffle=True,
                                                batch_size=8,
                                                input_nodes=('pheno', data['pheno'].train_mask))
                        
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)
                        
                        predicted_train = []
                        actual_train = []
                        @torch.no_grad()
                        def init_params():
                            batch = next(iter(train_loader))
                            batch = batch.to(device)
                            model(batch.x_dict, batch.edge_index_dict)
                        
                        
                        # ========================= Train the model ========================
                        def train(patience):
                            print('Combination ' + str(q+1))
                            
                            model.train()
                            optimizer = torch.optim.AdamW(model.parameters(), 
                                                         lr=comb[q][2], weight_decay=comb[q][1])
                        
                            loss_train_sum = 0
                            for epoch in range(comb[q][4]): 
                                if math.isinf(loss_train_sum) or math.isnan(loss_train_sum):# or loss_train_sum > 1000000:
                                    return 'invalid'
                                else:
                                    loss_train_sum = 0
                                for batch in train_loader:
                                    batch = batch.to(device)
                                    optimizer.zero_grad()
                                    out = model(batch.x_dict, batch.edge_index_dict)
                                    mask = batch['pheno'].train_mask
                                    loss = F.mse_loss(out['pheno'][mask], batch['pheno'].y[mask])
                                    loss_train_sum += loss
                                    loss.backward()
                                    optimizer.step()
                            print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')
                                
                            return model

                        #start = time.time()
                        m = train(np.inf)
                        #end = time.time()
                        
                    if type(m) is not str:
                        model = m
            
                        model.eval()
            
                        test_loader = HGTLoader(data, 
                                                num_samples=[100],
                                                batch_size=8,shuffle=False,
                                                input_nodes=('pheno', data['pheno'].test_mask))
            
                        predicted = []
                        actual = []
                        
                        for test in test_loader:
                            test = test.to(device)
                            result = model(test.x_dict, test.edge_index_dict)
                            predicted.append([item for sublist in result['pheno'][test['pheno'].test_mask].tolist() for item in sublist])
                            actual.append([item for sublist in test['pheno'].y[test['pheno'].test_mask].tolist() for item in sublist])
                                
                        predicted = [item for sublist in predicted for item in sublist]
                        actual = [item for sublist in actual for item in sublist]
                           
                        mse = mean_squared_error(actual,predicted)

                        r = scipy.stats.pearsonr(actual, predicted)[0]
         
                        r2 = r2_score(actual, predicted)
                        
                        result_pred = pd.concat([pd.DataFrame(predicted),
                                                 pd.DataFrame(actual)],axis=1)
                        result_pred.columns = ['predicted','actual']
                        result_pred['pop'] = pat[0]
                        result_pred['sample'] = sample+1
                        result_pred['id'] = 'NULL'
                        result_pred['phenotype'] = pat[4]
                        result_pred['repeat'] = num
                        
                        result_prediction_test = pd.concat([result_prediction_test, result_pred],axis=0)
                        
                        predicted_train = []
                        actual_train = []
                        result = model(data.x_dict, data.edge_index_dict)
                        predicted_train.append([item for sublist in result['pheno'][data['pheno'].train_mask].tolist() for item in sublist])
                        actual_train.append([item for sublist in data['pheno'].y[data['pheno'].train_mask].tolist() for item in sublist])
                        predicted_train = [k for i in predicted_train for k in i]
                        actual_train = [k for i in actual_train for k in i]

                        result_pred = pd.concat([pd.DataFrame(predicted_train),
                                                 pd.DataFrame(actual_train)],axis=1)
                        result_pred.columns = ['predicted','actual']
                        result_pred['pop'] = pat[0]
                        result_pred['sample'] = sample+1
                        result_pred['id'] = 'NULL'
                        result_pred['phenotype'] = pat[4]
                        result_pred['repeat'] = num
                        
                        result_prediction = pd.concat([result_prediction, result_pred],axis=0)
                        
                        if math.isnan(r) == False: 
                            print('Population : ' + str(pat[0]))
                            print('phenotype : ' + str(pat[4]))
                            print('column : ' + str(pat[3][yr]))
                            print('MSE : ' + str(mse))
                            print('Pearson r : ' + str(r))
                            print('R2 : ' + str(r2))
                            #print('time : ' + str(end-start))
               
                            tmp = pd.DataFrame([{'Population': str(pat[0]),
                                                  'Phenotype': pat[4],
                                                  'column':pat[3][yr],
                                                  'repeat': num,
                                                  'sample': sample,
                                                  'Pearson r':r,
                                                  'MSE':mse, 
                                                  'R2':r2,
                                                  #'time': end - start
                                                 }])
                            
                            record = pd.concat([record,tmp])

                            os.chdir(SAVE_PATH)                            
                            record.to_csv('GAT_pruned_0.8.csv')
                            result_prediction.to_csv('GAT_pruned_result_0.8.csv')
                            result_prediction_test.to_csv('GAT_pruned_test_result_0.8.csv')
                            num += 1
                            
            cnt_sample += 500                
                    
