
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

#import os

## Read data 
## Assume that each data has n lines(rows) & m-1 markers + 1 phenotype at the end(columns)
data_train = pd.read_csv('YOUR TRAIN DATA')
data_test = pd.read_csv('YOUR TEST DATA')

train_x, train_y = data_train.iloc[:,:-1], data_train.iloc[:,-1]
test_x, test_y = data_test.iloc[:,:-1], data_test.iloc[:,-1]

## Calculate the metrics
predicted = test_x.mean(axis=1)
actual_test = test_y.values.tolist()
mse = mean_squared_error(actual_test, predicted)
r = pearsonr(actual_test, predicted)[0]

## Store the metrics
record = pd.DataFrame([r,mse]).T
record.columns = ['Pearson_r','MSE']

## Store prediction result for the test data
result_prediction_test = pd.concat([pd.DataFrame(predicted),pd.DataFrame(actual_test)],axis=1)
result_prediction_test.columns = ['predicted','actual']         
         
## Store prediction result for the train data
predicted_train = train_x.mean(axis=1)
actual_train = train_y.values.tolist()
result_prediction_train = pd.concat([pd.DataFrame(predicted_train),pd.DataFrame(actual_train)],axis=1)
result_prediction_train.columns = ['predicted','actual']   

## Save all results
record.to_csv('METRIC DATA')
result_prediction_train.to_csv('PREDICTION RESULT FOR TRAIN DATA')
result_prediction_test.to_csv('PREDICTION RESULT FOR TEST DATA')


