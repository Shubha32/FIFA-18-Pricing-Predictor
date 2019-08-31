#XG BOOst

library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)


set.seed(123)

#final version ml package

#PCA FOR ML_PACKAGE VERSION 1.0-top 20 dimensions
dataset<-read.csv(file.choose())
dataset<-dataset[,-1]

#top 20 dimensions
k=20

#again trial

row = nrow(dataset)
col = ncol(dataset)

data1 = c()
for(i in 1:col) {
  data1 = c(data1,c(dataset[,i]))
}

data = matrix(data = data1,row,col)

data = data[,1:ncol(dataset)-1]

covMat = matrix(0,ncol(dataset)-1,ncol(dataset)-1)

for( i in 1:ncol(dataset)-1){
  for( j in 1:ncol(dataset)-1){
    covMat[i,j] = cov(data[,i],data[,j])
  }
}

eig = svd(covMat)

uReduce = eig$u[,1:k]

z = data %*% uReduce
z1 = cbind(z,dataset[,ncol(dataset)])
dat=z1
#dimensions now:15340 x 21

#10 fold cross validation:REGRESSION TREE

MyData<-dat
MyData<-MyData[sample(nrow(MyData)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(MyData)),breaks=10,labels=FALSE)

RMSEXGB=0
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- as.data.frame(MyData[testIndexes, ])
  trainData <- as.data.frame(MyData[-testIndexes, ])
  test<-as.data.frame(testData[,-ncol(testData)])
  
  trainm <- sparse.model.matrix(V21 ~ ., data = trainData)
  train_label<-trainData[,21]
  train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
  
  testm <- sparse.model.matrix(V20~., data = test)
  test_label<-testData[,ncol(testData)]
  test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)
  xgb_params <- list(
    "objective"="reg:linear")
  watch <- list(train = train_matrix, test = test_matrix)
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = 5000,
                         watchlist = watch,
                         eta = 0.02,
                         max.depth = 6,
                         gamma = 10,
                         subsample = 1,
                         colsample_bytree = 1,
                         missing = NA,
                         seed = 123
  )
  
  # Training & test error plot
  b <- data.frame(bst_model$evaluation_log)
  plot(b$iter, b$train_rmse, col = 'green')
  lines(b$iter, b$test_rmse, col = 'black')
  
  min(b$test_rmse)
  
  
  # Prediction & confusion matrix - test data
  p <- predict(bst_model, newdata = test_matrix)
  
  rmse2 <- sqrt(mean((test_label - p)^2))
  RMSEXGB=RMSEXGB+rmse2
}

print(RMSEXGB/10.0)