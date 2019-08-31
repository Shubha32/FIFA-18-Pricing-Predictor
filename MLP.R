#Trial 1.0 NN-ML_Package

library(mxnet)
data<-read.csv(file.choose(),header=TRUE)
sdata.x<-subset(data[,-1])
data.x<-subset(sdata.x[,-42])
View(data.x)
data.y<-data[,43]

ndiff_labels<-length(unique(data.y))
n_row<-nrow(data.x)

set.seed(123)
smp_size <- floor(0.7114732724902216 * nrow(data.x))
train_set <- sample(seq_len(nrow(data.x)), size = smp_size)
train.x <- data.matrix(data.x[train_set, ])
train.y<-data.y[train_set]
test.x<-data.matrix(data.x[-train_set,])
test.y<-data.y[-train_set]

# Define the input data
data = mx.symbol.Variable(name = 'data')

fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=41)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=2000)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=1500)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="relu")
fc4 <- mx.symbol.FullyConnected(act3, name="fc4", num_hidden=500)
act4 <- mx.symbol.Activation(fc4, name="relu4", act_type="relu")
fc5 <- mx.symbol.FullyConnected(act4, name="fc5", num_hidden=ndiff_labels)
softmax <- mx.symbol.SoftmaxOutput(fc5, name="sm")

devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, ctx=devices, num.round=10, array.batch.size=20,learning.rate=0.01, momentum=0.99,  eval.metric=mx.metric.accuracy,initializer=mx.init.uniform(0.07),epoch.end.callback=mx.callback.log.train.metric(100))

p <- predict(model, newdata = test.x)
mse <- sqrt(mean((test.y - p)^2))
