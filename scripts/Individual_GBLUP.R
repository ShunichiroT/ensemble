library(BGLR)
library(stringr)

nIter <- 12000
burnIn <- 2000
REPEAT <- 1
cnt <- 1
cnt_sample <- 500


pat <- list(list("W22TIL01",0,5,list(2,3),"DTA"),
                list("W22TIL01",0,5,list(6,7),"KW"),
                list("W22TIL01",0,5,list(10,11),"TILN"),
                list("W22TIL01",0,5,list(15),"PLHT"),
                list("W22TIL03",0,5,list(2,3),"DTA"),
                list("W22TIL03",0,5,list(6,7),"KW"),
                list("W22TIL03",0,5,list(10,11),"TILN"),
                list("W22TIL11",0,5,list(2,3),"DTA"),
                list("W22TIL11",0,5,list(6,7),"KW"),
                list("W22TIL11",0,5,list(10,11),"TILN"),
                list("W22TIL14",0,5,list(3,4),"DTA"),
                list("W22TIL14",0,5,list(7,8),"KW"),
                list("W22TIL14",0,5,list(11,12),"TILN"),
                list("W22TIL14",0,5,list(16),"PLHT"),
                list("W22TIL25",0,5,list(4,5),"DTA"),
                list("W22TIL25",0,5,list(8,9),"KW"),
                list("W22TIL25",0,5,list(12,13),"TILN"),
                list("W22TIL25",0,5,list(16,17),"PLHT"))

GENO_PATH <- "PATH"
PHENO_PATH <- "PATH"
SAVE_PATH <- "PATH"
IMP_PATH <- "PATH"

record <- data.frame(population=character(0),pheno=character(0),sample=numeric(0),column=numeric(0),rep=numeric(0),r=numeric(0),MSE=numeric(0))
result_prediction <- data.frame(predicted=numeric(0),actual=numeric(0),pop=character(0),sample=numeric(0),id=numeric(0),rep=numeric(0))
result_prediction_test <- data.frame(predicted=numeric(0),actual=numeric(0),pop=character(0),sample=numeric(0),id=numeric(0),rep=numeric(0))


setwd(IMP_PATH)
data_rf_imp <- read.csv('RF_SNPs_importance_avg_0.8_imputed.csv',check.names=FALSE)

setwd(IMP_PATH)
samples <- read.csv("samples_0.8.csv")
samples <- samples[,2:length(samples[1,])]


for (i in 1:length(samples[,1])){
  for (j in 1:length(samples[1,])){
    samples[[i,j]] <- str_sub(samples[i,j],2,-2)
    samples[[i,j]] <- list(as.numeric(unlist(strsplit(samples[[i,j]], ","))))
    samples[[j]][[i]][[1]] <- samples[[j]][[i]][[1]] + 1
  }
}

for (m in 1:length(pat)){
  if (m == 4 || m == 14){
    cnt <- cnt + 1
    next
  }else{
    if(pat[[m]][[5]]=='KW'){
      cnt_sample <- cnt_sample+ 500
      cnt <- cnt + 2
      next
    }
    for (p in 1:1){
      setwd(GENO_PATH)
      data1 <- read.csv(paste(pat[[m]][[1]],"_pruned_0.8_imputed.csv",sep=""),check.names=FALSE)[,-1]
      data2 <- read.csv(paste(pat[[m]][[1]],"_pruned_0.8_imputed.csv",sep=""),check.names=FALSE)[,-1]
      data <- rbind(data1,data2)
      
      setwd(PHENO_PATH)
      data_pheno <- read.csv(paste(pat[[m]][[1]],"_pheno.csv",sep=""))
      data_pheno1 <- data_pheno[, pat[[m]][[4]][[p]]]
      data_pheno2 <- data_pheno[, pat[[m]][[4]][[p+1]]]
      data_pheno <- append(data_pheno1, data_pheno2)
      
      delete <- c()
      data_rf_imp_target <- data.frame(t(data_rf_imp[c(cnt,cnt+1),])[2:length(t(data_rf_imp[cnt,])),])
      data_rf_imp_target <- data.frame(rowMeans(data_rf_imp_target))
      data_rf_imp_target['rows'] <- row.names(data_rf_imp_target)
      for (k in 1:length(row.names(data_rf_imp_target))){
        if (!(is.element(data_rf_imp_target[k,2], colnames(data)))){
          delete <- append(delete, k)
        }
      }
      
      data_rf_imp_target <- data_rf_imp_target[-c(delete),]
      data_rf_imp_target <- data.frame(data_rf_imp_target$rows[data_rf_imp_target$rowMeans.data_rf_imp_target. >= quantile(data_rf_imp_target$rowMeans.data_rf_imp_target., probs = 0.75)])
  
      data_columns <- colnames(data)
      delete <- c()
      cnt <- cnt + 2
      
      for (k in 1:length(data_columns)){
        if (!(is.element(data_columns[k], data_rf_imp_target[[1]]))){
          delete <- append(delete, k)
        }
      }
      
      data <- data[, -c(delete)]
      
      data['target'] <- data_pheno
      data <- data[is.na(data['target'])==FALSE,]
      data <- data[, colSums(is.na(data)) == 0]
      
      data_qtl <- data[,1:length(data[1,])-1]
      data_pheno <- data[,length(data[1,])]
      
      data1['target'] <- data_pheno1
      data1 <- data1[is.na(data1['target'])==FALSE,]
      data1 <- data1[, colSums(is.na(data1)) == 0]
      
      data2['target'] <- data_pheno2
      data2 <- data2[is.na(data2['target'])==FALSE,]
      data2 <- data2[, colSums(is.na(data2)) == 0]
      
      year1 = c()
      for (i in 1:length(data1[,1])){
        year1 = append(year1,0)
      }
      
      year2 = c()
      for (i in 1:length(data2[,1])){
        year2 = append(year2,1)
      }
      
      year = append(year1,year2)
      
      data_qtl['year'] <- year
      
      X=scale(data_qtl)/sqrt(ncol(data_qtl))
      y=data_pheno
      
      for (k in (cnt_sample-500+1):cnt_sample){
        
        test <- samples[[k,2]]
        train <- samples[[k,1]]
        
        y_test=y
        y_test[test[[1]]]=NA
        
        for (i in 1:REPEAT){
          fm=BGLR(y=y_test,ETA=list(mrk=list(X=X,model='BRR')),
                  nIter=nIter,burnIn=burnIn,saveAt='brr_'
          )
          
          y_predicted=fm$yHat[test[[1]]]
          y_actual = y[test[[1]]]
          
          #varE=scan('brr_varE.dat')
          #varU=scan('brr_ETA_mrk_varB.dat')
          #h2_1=varU/(varU+varE)
          
          pearson <- cor(y_predicted, y_actual, method = c("pearson"))
          pearson
          
          MSE <- mean((y_predicted - y_actual)^2)
          MSE
          
          pop <- rep(c(pat[[m]][[1]]), times=length(y_predicted))
          id <- test[[1]]
          ptype <- rep(c(pat[[m]][[5]]), times=length(y_predicted))
          repe <- rep(c(i), times=length(y_predicted))
          
          result_pred <- cbind(y_predicted, y_actual, pop, k, id, ptype, repe)
          result_prediction_test <- rbind(result_prediction_test, result_pred)
          
          
          y_predicted=fm$yHat[train[[1]]]
          y_actual = y[train[[1]]]
          
          pop <- rep(c(pat[[m]][[1]]), times=length(y_predicted))
          id <- train[[1]]
          ptype <- rep(c(pat[[m]][[5]]), times=length(y_predicted))
          repe <- rep(c(i), times=length(y_predicted))
          
          result_pred <- cbind(y_predicted, y_actual, pop, k, id, ptype, repe)
          
          result_prediction <- rbind(result_prediction, result_pred)
          
          record[nrow(record)+1,] = c(pat[[m]][[1]],pat[[m]][[5]],k,p,i,pearson,MSE)

          setwd(SAVE_PATH)
          write.csv(record, paste("GBLUP_0.8_imputed_concat.csv",sep=''), row.names=FALSE)
          write.csv(result_prediction, paste("GBLUP_0.8_imputed_concat_result.csv",sep=''), row.names=FALSE)
          write.csv(result_prediction_test, paste("GBLUP_0.8_imputed_test_concat_result.csv",sep=''), row.names=FALSE)
        }
        
      }
      cnt_sample <- cnt_sample + 500
    }
  }
}
