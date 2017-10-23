
library(data.table)
library(caret)
library(verification)

account_train <- fread("test data/raw_account_70_new.csv")
account_test <- fread("test data/raw_account_30_new.csv")

train <- fread("test data/raw_data_70_new.csv")
test <- fread("test data/raw_data_30_new.csv")

enquiry_train <- fread("test data/raw_enquiry_70_new.csv")
enquiry_test <- fread("test data/raw_enquiry_30_new.csv")


# Creating New Features 
Pay_history_train <- data.frame(customer_no=account_train$customer_no,h_length1=apply(account_train,2,nchar)[,13],h_length2=apply(account_train,2,nchar)[,14])
Pay_history_train$payment_history_mean_length	 <- rowMeans(Pay_history_train[-1],na.rm = TRUE)

pay_history_test <-data.frame(customer_no=account_test$customer_no, h_length1=apply(account_test,2,nchar)[,13],h_length2=apply(account_test,2,nchar)[,14])
pay_history_test$payment_history_mean_length	 <- rowMeans(pay_history_test[-1],na.rm = TRUE)

train <- merge(x = train, y = Pay_history_train[-2:-3], by = "customer_no", all = TRUE)
test <- merge(x = test, y = pay_history_test[-2:-3], by = "customer_no", all = TRUE)

names(train)

feature.names <- names(train)[2:ncol(train)]
feature.names<- feature.names[-82]

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


x_train <- train[,..feature.names]
y_train <- as.factor(train$Bad_label)

x_test <- test[,..feature.names]
y_test <- as.factor(test$Bad_label)

# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")

# normalized gini function 
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) 
    accum.losses <- temp.df$actual / total.losses 
    gini.sum <- cumsum(accum.losses - null.losses) 
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

# create the normalized gini summary function to pass into caret
giniSummary <- function (data, lev = "Yes", model = NULL) {
  levels(data$obs) <- c('0', '1')
  out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
  names(out) <- "NormalizedGini"
  out
}

# create the training control object. 
trControl = trainControl(
  method = 'cv',
  number = 2,
  summaryFunction = giniSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)

# create the tuning grid.
tuneGridXGB <- expand.grid(
  nrounds=c(50),
  max_depth = c(3,4),
  eta = c(0.05, 0.1),
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0))

start <- Sys.time()

# train the xgboost learner
xgbmod <- train(
  x = x_train,
  y = y_train,
  method = 'xgbTree',
  metric = 'NormalizedGini',
  trControl = trControl,
  tuneGrid = tuneGridXGB)


print(Sys.time() - start)

# make predictions
preds <- predict(xgbmod, newdata = x_test, type = "prob")

# convert test target values back to numeric for gini and roc.plot functions
levels(y_test) <- c("0", "1")
y_test_raw <- as.numeric(levels(y_test))[y_test]

# Diagnostics
print(xgbmod$results)
print(xgbmod$resample)

# plot results (useful for larger tuning grids)
plot(xgbmod)

# score the predictions against test data
normalizedGini(y_test_raw, preds$Yes)
# plot the ROC curve
roc.plot(y_test_raw, preds$Yes, plot.thres = c(0.02, 0.03, 0.04, 0.05))

