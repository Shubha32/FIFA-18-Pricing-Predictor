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

#10 fold cross validation: RANDOM FOREST

MyData<-dat
MyData<-MyData[sample(nrow(MyData)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(MyData)),breaks=10,labels=FALSE)


library(rpart)
library(caret)
library(randomForest)
set.seed(131)
RMSERF=0
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==1,arr.ind=TRUE)
  testData <- MyData[testIndexes, ]
  trainData <- MyData[-testIndexes, ]
  test<-as.data.frame(subset(testData[,-ncol(testData)]))
  
  model.rf <- randomForest(V21 ~ ., data=as.data.frame(trainData),
                           importance=TRUE,verbose=TRUE)
  print(model.rf)
  ## Show "importance" of variables: higher value mean more important:
  round(importance(model.rf), 2)
  
  ## "x" can be a matrix instead of a data frame:
  set.seed(17)
  p<-predict(model.rf,test)
  y<-testData[,ncol(testData)]
  rmse1 <- sqrt(mean((y - p)^2))
  RMSERF=RMSERF+rmse1
}

print(RMSERF/10.0)
