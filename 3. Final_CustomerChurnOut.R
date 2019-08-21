
#################################################################################################################
##############################  PREDICTING CUSTOMER CHURN OUTS ##################################################
#################################################################################################################

setwd('F:\\Deepak\\EPABA - IIMA\\Project')

#Reading customerdata
customerdata = read.csv('Project_Modeled_Data - Churn.csv')


######################## DATA PRE-PROCESSING ##########################

customerdata$State = factor(customerdata$State,
                            levels = c('Washington','Arizona','Nevada','Oregon','California'),
                            labels = c(1, 2, 3, 4, 5))

customerdata$Gender = factor(customerdata$Gender,
                             levels = c('M', 'F'),
                             labels = c(1,0))

customerdata$Vehicle.Size = factor(customerdata$Vehicle.Size,
                                   levels = c('Large','Medsize','Small'),
                                   labels = c(1,2,3))

customerdata$Policy = factor(customerdata$Policy,
                             levels = c('Corporate L1','Corporate L2','Corporate L3','Personal L1','Personal L2' ,'Personal L3', 'Special L1', 'Special L2', 'Special L3'),
                             labels = c(0, 1,2,3,4,5,6,7,8))

customerdata$Coverage = factor(customerdata$Coverage,
                               levels = c('Basic','Extended','Premium'),
                               labels = c(1,2,3))

customerdata$EmploymentStatus = factor(customerdata$EmploymentStatus, 
                                       levels = c('Employed','Unemployed','Disabled','Medical Leave','Retired'),
                                       labels = c(1,2,3,4,5))

customerdata$Response = factor(customerdata$Response,
                               levels = c('No','Yes'),
                               labels = c(0, 1))

customerdata$Education = factor(customerdata$Education,
                                levels = c('High School or Below', 'Bachelor','College', 'Master', 'Doctor' ),
                                labels = c(0,1,2,3,4))

customerdata$Location.Code = factor(customerdata$Location.Code,
                                    levels = c('Rural', 'Suburban', 'Urban'),
                                    labels = c(0,1,2))

customerdata$Marital.Status = factor(customerdata$Marital.Status,
                                     levels = c('Single','Married','Divorced'),
                                     labels = c(0, 1,2))

customerdata$Policy.Type = factor(customerdata$Policy.Type,
                                  levels = c('Corporate Auto', 'Personal Auto', 'Special Auto'),
                                  labels = c(0,1,2))

customerdata$Renew.Offer.Type = factor(customerdata$Renew.Offer.Type,
                                       levels = c('Offer1','Offer2','Offer3','Offer4'),
                                       labels = c(0,1,2,3))

customerdata$Sales.Channel = factor(customerdata$Sales.Channel,
                                    levels = c('Agent', 'Call Center', 'Branch','Web'),
                                    labels = c(0,1,2,3))

customerdata$Vehicle.Class = factor(customerdata$Vehicle.Class,
                                    levels = c('Two-Door Car','Four-Door Car','SUV','Luxury Car','Luxury SUV', 'Sports Car'),
                                    labels = c(0,1,2,3,4,5))

#### Splitting the data between train and test ######

library(caTools)
set.seed(101)
split <- sample.split(customerdata$Response, SplitRatio = 0.75)
trainingdata <- subset(customerdata, split == TRUE)
testdata <- subset(customerdata, split == FALSE)

##### Feature scaling Training Data Set  #####

trainingnum <- trainingdata[sapply(trainingdata,is.numeric)]
trainingnum <- scale(trainingnum)
trainingfactor <- trainingdata[sapply(trainingdata, is.factor)]
trainingfactor <- subset(trainingfactor, select = -c(Response))
trainingtarget <- trainingdata["Response"]
trainingfinal <- data.frame(trainingfactor, trainingnum, trainingtarget )

##### Feature Scaling Test Data Set #######

testnum <- testdata[sapply(testdata,is.numeric)]
testnum <- scale(testnum)
testfactor <- testdata[sapply(testdata, is.factor)]
testfactor <- subset(testfactor, select = -c(Response))
testtarget <- testdata["Response"]
testfinal <- data.frame(testfactor, testnum, testtarget )


###### Gini Coefficient FUNCTION #######################################
##### The below Code Snippet was obtained from R Library to Obtain Accuracy of the Statistical Models based on their Predictions ##### 

GiniCoefficient <- function(x, unbiased = TRUE, na.rm = FALSE){
  if (!is.numeric(x)){
    warning("'x' is not numeric; returning NA")
    return(NA)
  }
  if (!na.rm && any(na.ind <- is.na(x)))
    stop("'x' contain NAs")
  if (na.rm)
    x <- x[!na.ind]
  n <- length(x)
  mu <- mean(x)
  N <- if (unbiased) n * (n - 1) else n * n
  ox <- x[order(x)]
  dsum <- drop(crossprod(2 * 1:n - n - 1,  ox))
  dsum / (mu * N)
}


########################### LOGISTIC REGRESSION MODEL ###############################
#install.packages('ResourceSelection')
# install.packages('ROCR')

library(MASS)
library(MLmetrics)
library(ResourceSelection)
library(ROCR)
library(ggplot2)


regressor <- glm(formula = Response ~ ., 
                 family = binomial,
                 data = trainingfinal)
summary(regressor)
# Predicting the data on test data set
# This gives 87.64% Accuracy 

### Using stepAIC for Backward Elimination and Identifying all the appropriate Significant Variables ####
### Validating the Closely Correlated attributes by using StepAIC method (Backward elimination) ###

stepAIC(regressor, direction = 'both')


Stepregressor <- glm(formula = Response ~ Location.Code + Education + EmploymentStatus + 
              Marital.Status + Renew.Offer.Type + Vehicle.Size + Sales.Channel + 
              Months.Since.Last.Claim + Income + Total.Claim.Amount + Monthly.Premium.Auto + 
              Customer.Lifetime.Value, family = binomial, data = trainingfinal)

summary(Stepregressor)

hoslem.test(Stepregressor$y,Stepregressor$fitted.values,g=14)
# X-squared = 67.353, df = 12, p-value = 9.984e-10

ypredregressor <- predict(Stepregressor, type = 'response', newdata = testfinal)

# Checking the Accuracy for 0.5 Threshold
logregPredict  <- ifelse(ypredregressor > 0.5, 1, 0)

#### Confusion Matrix
cmlogreg <- table(Predicted = logregPredict, Actual = testdata$Response)
print(cmlogreg)
# Actual vs Predicted Accuracy = 87.03%
GiniCoefficient(c(1938,  63, 264, 18)) 

# Sensitivity  = 88.01%
1938 / (1938+264)

# Specificity = 77.78%
63/(18+63)

# Miscalculation Value = 12.352%
(1-sum(diag(cmlogreg))/sum(cmlogreg)) * 100 

## Checking Accuracy for 0.4 Threshold
logregPredict  <- ifelse(ypredregressor > 0.4, 1, 0)

#### Confusion Matrix
cmlogreg <- table(Predicted = logregPredict, Actual = testdata$Response)
print(cmlogreg)
# Actual vs Predicted Accuracy = 83.93%
GiniCoefficient(c(1908,  48, 248, 79))

# Miscalculation Value = 12.96%
(1-sum(diag(cmlogreg))/sum(cmlogreg)) * 100 

# Checking Accuracy for 0.6 Threshold
logregPredict  <- ifelse(ypredregressor > 0.6, 1, 0)

#### Confusion Matrix
cmlogreg <- table(Predicted = logregPredict, Actual = testdata$Response)
print(cmlogreg)
# Actual vs Predicted Accuracy = 87.76%
GiniCoefficient(c(1945,  11, 268, 59))

# MISCALCULATION VALUE / PERCENTAGE = 12.22%
(1-sum(diag(cmlogreg))/sum(cmlogreg)) * 100

# Plotting FOur Fold Plot for Confusion Matrix
ctable <- as.table(matrix(c(1945,  11, 268, 59), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

# WIth GINI Co-Efficients this model gives us 87.76 % Accuracy
GiniCoefficient(c(1945,  11, 268, 59)) 

# Specificity = TRUE Negative Rate = 0.8428 = 84.28%
59 / (11+59)  

# False Positive Rate  =  1 - Specificity = 0.1571 = 15.71%
1 - (59/(11+59)) = 11 / (11+59)

# Sensitivity = TRUE POSITIVE RATE = 0.8788 = 87.88%
1945 / (1945 + 268) 

# FALSE NEGATIVE RATE = 1 - Sensitivity = 0.121 = 12.1%
268 / (1945 + 268) 

#######  ROC CURVE ##########

ROCPredict <-prediction(logregPredict ,testdata$Response)
ROCPerf  = performance(ROCPredict, "tpr", "fpr")
plot(ROCPerf)


############# KERNEL SVM  ########################### 

library(e1071)

# Using All the Variables, prediction accuracy is reducing and hence, using the already identified Significant variables to apply SVM


classifier <- svm(formula = Response ~ Location.Code + Education + EmploymentStatus + 
                    Marital.Status + Renew.Offer.Type + Vehicle.Size + Sales.Channel + 
                    Months.Since.Last.Claim + Income + Total.Claim.Amount + Monthly.Premium.Auto + 
                    Customer.Lifetime.Value, 
                  data = trainingfinal,
                  type = 'C-classification',
                  Kernel = 'polynomial')

#Predicting the data
SVMPredict <- predict(classifier, newdata = testfinal)

cmksvm <- table(Predicted = SVMPredict, Actual = testdata$Response )
print(cmksvm)

# Plotting Four Fold Plot
ctable <- as.table(matrix(c(1942, 14, 262, 65), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

# Gini Co-efficients  = 87.32%
GiniCoefficient(c(1942, 14, 262, 65))

#######  ROC CURVE ##########

ROCPredict <-prediction(SVMPredict ,testdata$Response)
ROCPerf  = performance(ROCPredict, "tpr", "fpr")
plot(ROCPerf)


# Specificity = TRUE Negative Rate = 0.8227848 = 82.27%
65 / (65+14)  

# False Positive Rate  =  1 - Specificity = 0.1772152 = 17.72%
14 / (14+65)

# Sensitivity = TRUE POSITIVE RATE = 0.8811252 = 88.11%
1942 / (1942 + 262) 

# FALSE NEGATIVE RATE = 1 - Sensitivity = 0.1188748 = 11.88%
262 / (1942 + 262) 

# MISCALCULATION VALUE / PERCENTAGE = 12.08%
(1-sum(diag(cmksvm))/sum(cmksvm)) * 100


###################### NAIVE BAYES ALGORITHM #######################################

library(e1071)


Nclassifier <- naiveBayes(x = trainingfinal[,-22], 
                          y = trainingfinal$Response)

# Predicting for TestSet
NaiveBpredict <- predict(Nclassifier, newdata = testfinal[,-22])

cmNB <- table(Predict = NaiveBpredict, Actual = testdata$Response )
print(cmNB)

# Plotting Four Fold graph on COnfusion Matrix
ctable <- as.table(matrix(c(1936, 20, 261, 66), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

# Gini Co-efficients = 0.8677 = 86.77%
GiniCoefficient(c(1936, 20, 261, 66))

# Specificity = TRUE Negative Rate = 0.7674419 = 76.74%
66 / (66+20)  

# False Positive Rate  =  1 - Specificity = 0.23255 = 23.25%
20 / (20+66)

# Sensitivity = TRUE POSITIVE RATE = 0.8811252 = 88.12%
1936 / (1936 + 261) 

# FALSE NEGATIVE RATE = 1 - Sensitivity = 0.1187 = 11.87%
261 / ( 1936 + 261) 

# MISCALCULATION VALUE / PERCENTAGE = 12.30%
(1-sum(diag(cmNB))/sum(cmNB)) * 100


#################### CONCLUSION ###########################
##### KERNEL SVM HAS A BETTER PERFORMANCE OVER OTHER REGRESSION MODEL TO PREDICT THE CUSTOMER CHURN ######



