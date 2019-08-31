#PCA FOR ML_PACKAGE VERSION 1.0
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


mse2 <- sqrt(mean((y - p)^2))
