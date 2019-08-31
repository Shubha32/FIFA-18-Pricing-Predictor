#PCA FOR ML_PACKAGE VERSION 1.0 : without cross validation
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
#Regression tree

library(rpart)

#Create train and test sets

set.seed(123)
smp_size <- floor(0.7114732724902216 * nrow(dat))
train_set <- sample(seq_len(nrow(dat)), size = smp_size)
train <- subset(dat[train_set, ])

train_fin<-as.data.frame(train)

library(caret)

#Random Forest

library(randomForest)
set.seed(131)
model.rf <- randomForest(V21 ~ ., data=train_fin,
                         importance=TRUE,verbose=TRUE)
print(model.rf)
## Show "importance" of variables: higher value mean more important:
round(importance(model.rf), 2)

## "x" can be a matrix instead of a data frame:
set.seed(17)
p<-predict(model.rf,test)

mse1 <- sqrt(mean((y - p)^2))


