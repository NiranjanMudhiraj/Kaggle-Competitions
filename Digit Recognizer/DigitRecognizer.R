
library(caret)
library(e1071)
library(kernlab)

train_data = read.csv("train.csv")

dim(train_data)
str(train_data)

train_data_label = as.factor(train_data$label)
train_data$label=NULL

variance_data = nearZeroVar(train_data)
length(variance_data)
variance_data_rm = train_data[,-variance_data]
dim(variance_data_rm)

pp_nor_pca = preProcess(variance_data_rm,method = c("range"))
#original data is being transformed into new dimensions
predict_pca = predict(pp_nor_pca,variance_data_rm)
dim(predict_pca)
class(predict_pca)

filter <- ksvm(train_data_label~.,data=predict_pca,kernel="rbfdot",
               kpar="automatic",C=5,cross=3)
filter




#**************************  TEST DATA *************

test_data = read.csv("test.csv")
dim(test_data)
str(test_data)

variance_data_rm_t = test_data[,-variance_data]
dim(variance_data_rm_t)


test_data_pca = predict(pp_nor_pca,newdata=variance_data_rm_t)

class(test_data_pca)

result_svm = predict(filter, newdata = test_data_pca)


#****************  Submission  ******************

prediction.table1 <- data.frame(ImageId=1:nrow(test_data), Label=result_svm)
write.csv(prediction.table1,file = "submission_svm_g2.csv",row.names = FALSE)

