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


library(rpart)
RMSERT=0
#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- MyData[testIndexes, ]
  trainData <- MyData[-testIndexes, ]
  
  fit <- rpart(V21 ~ ., 
               method="anova", data=as.data.frame(trainData))
  
  plot(fit, uniform=TRUE, 
       main="Regression Tree for Our Data")
  text(fit, use.n=TRUE, cex = .6)
  
  
  printcp(fit)
  
  #what are these??
  par(mfrow=c(1,2)) 
  rsq.rpart(fit)
  
  test<-as.data.frame(subset(testData[,-ncol(testData)]))
  y<-testData[,ncol(dat)]
  
  #what is this???
  predictions<-predict(fit, test, method = "anova")
  
  rmse <- sqrt(mean((y - predictions)^2))
  RMSERT=RMSERT+rmse
}

print(RMSERT/10.0)
#Regression tree


#Create train and test sets

smp_size <- floor(0.7114732724902216 * nrow(dat))
train_set <- sample(seq_len(nrow(dat)), size = smp_size)
train <- subset(dat[train_set, ])

train_fin<-as.data.frame(train)


fit <- rpart(V21 ~ ., 
             method="anova", data=train_fin )
plot(fit, uniform=TRUE, 
     main="Regression Tree for Our Data")
text(fit, use.n=TRUE, cex = .6)


printcp(fit)

#what are these??
par(mfrow=c(1,2)) 
rsq.rpart(fit)

test<-as.data.frame(subset(dat[-train_set,-ncol(dat)]))
y<-dat[-train_set,ncol(dat)]

#what is this???
predictions<-predict(fit, test, method = "anova")

mse <- mean((y - predictions)^2)

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

mse1 <- mean((y - p)^2)


#XG BOOst

library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)


set.seed(123)


#Splitting data



# Creating sparse matrix
trainm <- sparse.model.matrix(V21 ~ ., data = train_fin)
head(trainm)
train_label<-train_fin[,21]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

testm <- sparse.model.matrix(V20~., data = test)
test_label<-dat[-train_set,ncol(dat)]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

testm
# Parameters
#unique(train_label)
nc <- length(unique(train_label))
xgb_params <- list(
  "objective"="reg:linear")
watch <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 8000,
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


mse2 <- mean((y - p)^2)
