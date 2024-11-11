library(BGLR)
library(stringr)
library(data.table)
library(dplyr)
library(iml)


## Read data 
## Assume that each data is structured as below:
## Rows: n RILs(n rows in total) 
## columns: the first column for id, m columns for markers and the last column for phenotype(m+2 columns in total)
setwd('your/path/ensemble-main/code')
data_train <- fread("../data/example_train.csv",header=TRUE)
data_test <- fread("../data/example_test.csv",header=TRUE)
tr_id <- data_train[,1]
te_id <- data_test[,1]
data <- rbind(data_train[,2:ncol(data_train)],data_test[,2:ncol(data_test)])

## Preprocess the data to be processed by the model
data_qtl <- data.frame(lapply(data[,1:(length(data[1,])-1)], as.numeric))
data_pheno <- data[,length(data[1,]):length(data[1,])]

D <- as.matrix(dist(data_qtl,method="euclidean"))^2
D <- D/mean(D)
h <- 1
K <- exp(-h*D)

y <- as.numeric(unlist(data_pheno))

## Assign NA to the test data for evaluation
train_id <- seq(1,nrow(data_train))
test_id <- seq((nrow(data_train)+1),(nrow(data_train)+nrow(data_test)))

y_test <- y
y_test[test_id] <- NA

## Predict train data phenotypes by the model
nIter <- 12000
burnIn <- 2000

fm <- BGLR(y=y_test,ETA=list(list(K=K,model='RKHS')),
           nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt='eig_')

y_predicted <- fm$yHat[test_id]
y_actual = y[test_id]

## Calculate the metrics
pearson <- cor(y_predicted, y_actual, method = c("pearson"))
MSE <- mean((y_predicted - y_actual)^2)

record <- data.frame(cbind(pearson,MSE))
colnames(record) <- c('Pearson_r','MSE')

## Store prediction result for the test data
result_prediction_test <- cbind(te_id,y_predicted,y_actual)
colnames(result_prediction_test) <- c('id','predicted','actual')

## Store prediction result for the train data
y_predicted_train <- fm$yHat[train_id]
y_actual_train <- y[train_id]

result_prediction_train <- cbind(tr_id, y_predicted_train,y_actual_train)
colnames(result_prediction_train) <- c('id','predicted','actual')

## Extract marker effect
true_model <- function(newdata) {
  qtl <- rbind(data_qtl,newdata)
  pred <- c(y_test,rep(NA, nrow(newdata)))
  len_beg <- nrow(data_qtl)+ 1
  len_end <- nrow(data_qtl)+ nrow(newdata)
  
  D <- as.matrix(dist(qtl,method="euclidean"))^2
  D <- D/mean(D)
  h <- 1
  K <- exp(-h*D)

  f <- BGLR(y=pred,ETA=list(list(K=K,model='RKHS')),
             nIter=nIter,burnIn=burnIn,verbose=FALSE,saveAt='eig_')
  
  return(f[["yHat"]][len_beg:len_end])
}

predictor <- Predictor$new(NULL, data = data_qtl, y=fm[["yHat"]], predict.fun=true_model)

effect <- data.frame()
if(length(test_id) < 50){len <- length(test_id)}else{len <- 50}
for(j in 1:len){
  shapley <- Shapley$new(predictor, x.interest = data_qtl[test_id[j],], sample.size = 1)
  tmp <- data.frame(t(shapley$results[,1:2]))
  colnames(tmp) <- colnames(data)[1:ncol(data)-1]
  effect <- dplyr::bind_rows(effect, tmp[2,])
}
colnames(effect) <- colnames(data)[1:ncol(data)-1]

## Save all results
write.csv(record, "METRIC_RKHS.csv")
write.csv(effect, "MARKER_EFFECT_RKHS.csv")
setwd('your/path/ensemble-main/output')
write.csv(result_prediction_train, "Prediction_result_train_RKHS.csv")
write.csv(result_prediction_test, "Prediction_result_test_RKHS.csv")