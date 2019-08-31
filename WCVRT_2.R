#PCA FOR ML_PACKAGE VERSION 1.0 - without cross validation

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


fit <- rpart(V21 ~ ., 
             method="anova", data=train_fin )
plot(fit, uniform=TRUE, 
     main="Regression Tree for Our Data")
text(fit, use.n=TRUE, cex = .6)


printcp(fit)

par(mfrow=c(1,2)) 
rsq.rpart(fit)

test<-as.data.frame(subset(dat[-train_set,-ncol(dat)]))
y<-dat[-train_set,ncol(dat)]

#what is this???
predictions<-predict(fit, test, method = "anova")

rmse <- sqrt(mean((y - predictions)^2))
cat("Without cross validation:",rmse)