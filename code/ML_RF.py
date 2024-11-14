
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import os


## Read data 
## Assume that each data is structured as below:
## Rows: n RILs(n rows in total) 
## columns: the first column for id, m columns for markers and the last column for phenotype(m+2 columns in total)
data_train = pd.read_csv('../data/example_train.csv')
data_test = pd.read_csv('../data/example_test.csv')

train_id, test_id = data_train.iloc[:,0], data_test.iloc[:,0]
train_x, train_y = data_train.iloc[:,1:-1], data_train.iloc[:,-1]
test_x, test_y = data_test.iloc[:,1:-1], data_test.iloc[:,-1]

## Train the model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)
rf.fit(train_x, train_y);
     
## Predict phenotypes for the test data
predicted = rf.predict(test_x)

## Calculate the metrics
actual_test = test_y.values.tolist()
mse = mean_squared_error(actual_test, predicted)
r = pearsonr(actual_test, predicted)[0]

## Store the metrics
record = pd.DataFrame([r,mse]).T
record.columns = ['Pearson_r','MSE']

## Store prediction result for the test data
result_prediction_test = pd.concat([pd.DataFrame(test_id), pd.DataFrame(predicted),pd.DataFrame(actual_test)],axis=1)
result_prediction_test.columns = ['id','predicted','actual']         
         
## Store prediction result for the train data
predicted_train = rf.predict(train_x)
actual_train = train_y.values.tolist()
result_prediction_train = pd.concat([pd.DataFrame(train_id), pd.DataFrame(predicted_train),pd.DataFrame(actual_train)],axis=1)
result_prediction_train.columns = ['id','predicted','actual']   

## Extract marker effect
effect = pd.DataFrame(rf.feature_importances_).T
effect.columns = list(data_train.columns)[1:-1]

## Save all results
record.to_csv('../output/Metric_RF.csv')
effect.to_csv('../output/Marker_effect_RF.csv')
result_prediction_train.to_csv('../output/Prediction_result_train_RF.csv')
result_prediction_test.to_csv('../output/Prediction_result_test_RF.csv')
