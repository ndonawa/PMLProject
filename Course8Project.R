library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
library(lubridate)
library(forecast)
library(e1071)
library(elasticnet)
library(dplyr)
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

##Explore Data - Identify NAs
dim(training)
summary(training)
nrows = nrow(training)
ncomplete = sum(complete.cases(training))
ncomplete / nrows ## Only 2% of the rows have complete data. 

##Find columns with NAs and remove
count_NA<- sapply(testing, function(y) sum((is.na(y))))
NA_values <- count_NA[count_Na == 20]
var_remove <- names(NA_values)

cleanTraining <-training[,!(names(training) %in% var_remove)]
cleanTesting <- testing[,!(names(training) %in% var_remove)]
##Create Training validation Set

cleanTraining <- cleanTraining[c(-1, -2,-3,-4,-5,-6,-7)]
cleanTesting <- cleanTesting[c(-1, -2,-3,-4,-5,-6,-7)]
dim(cleanTraining)
## Validation Set 
set.seed(1021)
tsTraining <- createDataPartition(y = cleanTraining$classe, 
                                  p =.7,
                                  list = FALSE)
validationTraining <- cleanTraining[tsTraining, ]
validationTesting <- cleanTraining[-tsTraining, ]

##Train Model with prediction Trees
trControl <- trainControl(method = "cv", number =5)
modFit <- train(classe ~., method = "rpart", data = cleanTraining, trControl = trControl)
print(modFit$finalModel)
plot(modFit$finalModel, uniform = TRUE,
     main ="Classification Tree")
text(modFit$finalModel, use.n = TRUE, all = TRUE, cex=.8)
##Improve Plot
library(rattle)
fancyRpartPlot(modFit$finalModel)

##Predict New Values
pred1 <- predict(modFit, newdata = validationTesting)
matrix <- confusionMatrix(validationTesting$classe, pred1)
matrix$overall[1]

##Train Model with Random Forests
modFit1 <- train(classe~., data = cleanTraining, method = "rf", trControl = trControl, verbose = FALSE)
print(modFit1)
plot(modFit1, main = "Model Accuracy ~ Predictors")
##Predict new values
pred2 <- predict(modFit1, newdata = validationTesting)
matrix1 <- confusionMatrix(validationTesting$classe, pred2)
matrix1$overall[1]

##Model Error
plot(modFit1$finalModel, main = "Random Forest Error ~ Tree Number")


##Final Prediction
pred3 <- predict(modFit1, newdata = cleanTesting)
pred3