library(BGLR)
library(stringr)
library(data.table)
library(dplyr)

setwd('C:/Users/uqstomur/OneDrive - The University of Queensland/Documents/Data/TeoNAM')

## Read data 
## Assume that each data has n lines(rows) & m-1 markers + 1 phenotype at the end(columns)
data_train <- fread("example_G3_train.csv",header=TRUE)
data_test <- fread("example_G3_test.csv",header=TRUE)
data <- rbind(data_train,data_test)

## Preprocess the data to be processed by the model
data_qtl <- data.frame(lapply(data[,1:(length(data[1,])-1)], as.numeric))
data_pheno <- data[,length(data[1,]):length(data[1,])]

X <- scale(data_qtl)/sqrt(ncol(data_qtl))
y <- as.numeric(unlist(data_pheno))

## Assign NA to the test data for evaluation
train_id <- seq(1,nrow(data_train))
test_id <- seq((nrow(data_train)+1),(nrow(data_train)+nrow(data_test)))

y_test <- y
y_test[test_id] <- NA

## Predict train data phenotypes by the model
nIter <- 12000
burnIn <- 2000

fm <- BGLR(y=y_test,ETA=list(mrk=list(X=X,model='BayesB')),
           nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt='bayes_')

y_predicted <- fm$yHat[test_id]
y_actual = y[test_id]

## Calculate the metrics
pearson <- cor(y_predicted, y_actual, method = c("pearson"))
MSE <- mean((y_predicted - y_actual)^2)

record <- data.frame(cbind(pearson,MSE))
colnames(record) <- c('Pearson_r','MSE')

## Store prediction result for the test data
result_prediction_test <- cbind(y_predicted,y_actual)
colnames(result_prediction_test) <- c('predicted','actual')

## Store prediction result for the train data
y_predicted_train <- fm$yHat[train_id]
y_actual_train <- y[train_id]

result_prediction_train <- cbind(y_predicted_train,y_actual_train)
colnames(result_prediction_train) <- c('predicted','actual')

## Extract marker effect
effect <- data.frame(t(fm[["ETA"]][["mrk"]][["b"]]))
colnames(effect) <- colnames(data)[1:ncol(data)-1]

## Save all results
write.csv(record, "METRIC DATA")
write.csv(effect, "MARKER EFFECT DATA")
write.csv(result_prediction_train, "PREDICTION RESULT FOR TRAIN DATA")
write.csv(result_prediction_test, "PREDICTION RESULT FOR TEST DATA")

