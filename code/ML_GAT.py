
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, Linear, to_hetero_with_bases
from torch_geometric.loader import HGTLoader
from torch_geometric.explain import Explainer, CaptumExplainer


## Read data 
## Assume that each data is structured as below:
## Rows: n RILs(n rows in total) 
## columns: the first column for id, m columns for markers and the last column for phenotype(m+2 columns in total)
data_train = pd.read_csv('../data/example_train.csv')
data_test = pd.read_csv('../data/example_test.csv')

## Preprocess the data so that it caon be converted into a graph format
tr_id, te_id = data_train.iloc[:,0], data_test.iloc[:,0]
data = pd.concat([data_train,data_test])
data_QTL, data_pheno = data.iloc[:,1:-1].reset_index(drop=True), data.iloc[:,-1].reset_index(drop=True)

x_qtl = []
for i in range(len(data_QTL.iloc[0,:])):
    x_qtl.append(list(map(lambda x: [x], data_QTL.iloc[:,i].to_list())))
x_qtl = np.array(x_qtl)

x_pheno = [[0]] * len(data)    
y_pheno = np.array(list(map(lambda x: [x], data_pheno.to_list())), dtype='float32')

edges = np.array(range(0,len(data_QTL.iloc[:,0])))

## Convert the data into a graph
data = HeteroData()
data['pheno'].x = torch.tensor(x_pheno, dtype=torch.float)
data['pheno'].y = torch.from_numpy(y_pheno)
 
for i in range(len(x_qtl)):
    data['qtl_'+str(i+1)].x = torch.tensor(x_qtl[i], dtype=torch.float)  
    data['qtl_'+str(i+1),'affect','pheno'].edge_index = torch.stack([torch.from_numpy(edges).to(torch.long),torch.from_numpy(edges).to(torch.long)], dim=0)

## Add masks to distinguish between train and test data
ids = list(range(data_pheno.shape[0]))
train_id = list(range(data_train.shape[0]))
test_id = list(range(data_train.shape[0], (data_train.shape[0]+data_train.shape[1])))

mask_train = np.array([])
mask_valid = np.array([])
mask_test = np.array([])
for i in range(len(ids)):
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

## set hyperparameters
regularisation = 5e-4 
learning_rate = 0.005
dropout = 0.9
epoch = 150
channels = 20
heads = 8

## Define GAT class
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dpout):
        super().__init__()
        self.conv1 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dpout)
        self.lin1 = Linear(-1, hidden_channels*heads)
        self.conv2 = GATv2Conv((-1,-1), hidden_channels, add_self_loops=False,heads=heads, concat=True, dropout=dpout)
        self.lin2 = Linear(-1, hidden_channels*heads)
        self.conv3 = GATv2Conv((-1,-1), out_channels, add_self_loops=False, heads=heads, concat=False, dropout=dpout)
        self.lin3 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.elu(x)
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x
    
class NodeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

    def forward(self, z_dict):
        return z_dict['pheno'].mean(axis=1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels,out_channels, dpout):
        super().__init__()
        self.encoder = GAT(hidden_channels, out_channels,dpout)
        self.encoder = to_hetero_with_bases(self.encoder, data.metadata(), num_bases=5)
        self.decoder = NodeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict)

## Define training pipeline
def train():
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=regularisation)
    
    for ep in range(epoch):        
        loss_train_sum = 0
        for batch in train_loader:
            batch = batch.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            mask = batch['pheno'].train_mask
            loss = F.mse_loss(out[mask].unsqueeze(-1), batch['pheno'].y[mask])
            loss_train_sum += loss
            loss.backward()
            optimizer.step()
        print(f'Epoch {ep:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')
    
    return model
             

## Develop a GAT model
model = Model(hidden_channels=channels, out_channels=1, dpout=dropout)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)

## Convert the data into mini-batches
train_loader = HGTLoader(data, 
                        num_samples={key:[1064] * 4 for key in data.node_types},
                        shuffle=True,
                        batch_size=8,
                        input_nodes=('pheno', data['pheno'].train_mask))
test_loader = HGTLoader(data, 
                         num_samples={key:[1064] * 4 for key in data.node_types},
                         batch_size=8,shuffle=False,
                         input_nodes=('pheno', data['pheno'].test_mask))
## Train a model
model = train()

## Predict phenotypes for the test data
model.eval()
predicted_test = []
actual_test = []
for test in test_loader:
    test = test.to(device)
    result = model(test.x_dict, test.edge_index_dict)
    predicted_test.append(result[test['pheno'].test_mask].tolist())
    actual_test.append([item for sublist in test['pheno'].y[test['pheno'].test_mask].tolist() for item in sublist])
predicted_test = [item for sublist in predicted_test for item in sublist]
actual_test = [item for sublist in actual_test for item in sublist]

## Calculate the metrics
mse = mean_squared_error(actual_test,predicted_test)
r = pearsonr(actual_test, predicted_test)[0]

## Store the metrics
record = pd.DataFrame([r,mse]).T
record.columns = ['Pearson_r','MSE']

## Store prediction result for the test data
result_prediction_test = pd.concat([pd.DataFrame(te_id), pd.DataFrame(predicted_test),pd.DataFrame(actual_test)],axis=1)
result_prediction_test.columns = ['id','predicted','actual']  

## Store prediction result for the train data
predicted_train = []
actual_train = []
result = model(data.x_dict, data.edge_index_dict)
predicted_train.append(result[data['pheno'].train_mask].tolist())
actual_train.append([item for sublist in data['pheno'].y[data['pheno'].train_mask].tolist() for item in sublist])
predicted_train = [k for i in predicted_train for k in i]
actual_train = [k for i in actual_train for k in i]
result_prediction_train = pd.concat([pd.DataFrame(tr_id), pd.DataFrame(predicted_train),pd.DataFrame(actual_train)],axis=1)
result_prediction_train.columns = ['id','predicted','actual']  

## Extract marker effects
bline = tuple([0] * len(data.x_dict))
explainer = Explainer(
    model = model,
    algorithm=CaptumExplainer('IntegratedGradients',baselines=bline),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None, # do not change here
    model_config = dict(
        mode='regression',
        task_level='node',
        return_type='raw',
        ),
    threshold_config=dict(threshold_type='topk', value=10),
)

hetero_explanation = explainer(
    data.x_dict,
    data.edge_index_dict,
)

effect = pd.DataFrame()
for ii in range(1,len(data.x_dict)):
    effect = pd.concat([effect, pd.DataFrame(hetero_explanation['qtl_'+str(ii)]['node_mask'].squeeze().tolist())],axis=1)
effect.columns = list(data_QTL.columns) 

## Save all results
record.to_csv('Metric_GAT.csv')
effect.to_csv('Marker_effect_GAT.csv')
result_prediction_train.to_csv('../output/Prediction_result_train_GAT.csv')
result_prediction_test.to_csv('../output/Prediction_result_test_GAT.csv')
