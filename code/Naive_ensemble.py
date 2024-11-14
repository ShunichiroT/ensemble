
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


## Read data 
## Assume that each data is structured as below:
## Rows: n RILs(n rows in total) 
## columns: the first column for id, next six columns for the predicted phenotypes of each prediction model and the last column for phenotype(eight columns in total)
#data_train = pd.read_csv('../data/example_train.csv')
data_test = pd.read_csv('../data/example_matrix.csv')

#train_x, train_y = data_train.iloc[:,:-1], data_train.iloc[:,-1]
#train_id = data_train.iloc[:,0]
test_id = data_test.iloc[:,0]
test_x, test_y = data_test.iloc[:,1:-1], data_test.iloc[:,-1]

## Calculate the metrics
predicted = test_x.mean(axis=1)
actual_test = test_y.values.tolist()
mse = mean_squared_error(actual_test, predicted)
r = pearsonr(actual_test, predicted)[0]

## Store the metrics
record = pd.DataFrame([r,mse]).T
record.columns = ['Pearson_r','MSE']

## Store prediction result for the test data
result_prediction_test = pd.concat([pd.DataFrame(test_id), pd.DataFrame(predicted),pd.DataFrame(actual_test)],axis=1)
result_prediction_test.columns = ['id','predicted','actual']         
         
## Store prediction result for the train data if necessary
#predicted_train = train_x.mean(axis=1)
#actual_train = train_y.values.tolist()
#result_prediction_train = pd.concat([pd.DataFrame(train_id), pd.DataFrame(predicted_train),pd.DataFrame(actual_train)],axis=1)
#result_prediction_train.columns = ['id','predicted','actual']   

## Save all results
record.to_csv('../output/Metric_ensemble.csv')
#result_prediction_train.to_csv('../output/Prediction_result_test_ensemble.csv')
result_prediction_test.to_csv('../output/Prediction_result_test_ensemble.csv')


