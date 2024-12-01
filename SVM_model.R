rm(list = ls())

df <- read.csv("C:\\Users\\nehas\\Documents\\GT 2023\\MGT 6203\\Project\\bank-full.csv", sep=';')

library(dplyr)
library(tidyverse)
library(scales)
library(caret)
library(ROSE)
library(e1071)
library(corrplot)

#-----------------------------------------------------------------------------------------------------------

#CLEANING AND DATA TRANSFORMATION
#check for missing values
colSums(is.na(df))
#from the output we can see that there are no NULL values for any attribute 

#however there are a lot of unknown values 
df %>%
  summarise_all(list(~sum(. == "unknown")))

#Since over half the data has unknowns for poutcome, we will remove the entire column (not useful in predicting closing probability)
#There is quite a alot of unknows for contact method as well, remove entire column (not using this variable to predict closing probability )
df = subset(df, select = -c(poutcome, contact))
#There are also a few unknowns for job and education
##we will simply remove these rows (288+ 1875)with unknowns
##Note that deleting rows/data for missing data results in data loss and may introduce potential bias
##But in this case we have well over 45,000 observations and removing about 1200 should not bias the data
table(df$y)
##original data has 13.24 success rate 
df <- df[df$job != "unknown" & df$education != "unknown", ]
table(df$y)
##after removal of rows we still have 13.15% success rate 
##so we are nto really biasing the data, difference is only 0.09% so we should be fine to move forward 

#we will also remove duration column because duration of a call is an aspect that cannot be decided before calling customers
df = subset(df, select = -c(duration))

#CORRELATION MATRIX FOR NUMERIC ATTRIBUTES
par(mfrow=c(1,1))
dfcor <- cor(df[,c(1,6,9,11,12,13)], 
             method = "spearman")

corrplot(dfcor,
         method = "color",
         tl.cex = 0.9,
         number.cex = 0.95,
         addCoef.col = "black")
#we can see there is a strong correlation between previous and pdays.
#so we need to remove one
as.data.frame(table(df$previous))
as.data.frame(table(df$pdays))
df = subset(df, select = -c(pdays))
#after looking at a frequency tabel, i removed pdays since previous may hold valuable data in relationship to our problem 

#OUTLIERS FOR RELEVANT NUMERIC ATTRIBUTES 
#unique values 
apply(df,2,function(x) length(unique(x)))
#boxplots
par(mfrow=c(2,2))
boxplot(df$balance, xlab = "Balance")
boxplot(df$campaign, xlab = "Campaign")
boxplot(df$previous, xlab = "Previous")
# it seems like there are a lot of outliers for each numeric attribute, especially Balance
# however we cannot say for sure they are outliers without other conext 
summary(df$balance)
par(mfrow=c(1,1))
ggplot(df,aes(x = df$balance)) + geom_histogram(aes(y = after_stat(count / sum(count))), color="black", fill="light blue", binwidth = 500) + 
  labs(title = "Balance Amount Distribution") + xlim(-1000,10000) +
  stat_bin(
    binwidth = 500, geom = "text", color = "black",
    aes(y = after_stat(count / sum(count)), 
        label = scales::percent(after_stat(count / sum(count)))),
    position = position_stack(vjust = 0.9)
  )+
  scale_y_continuous(labels = scales::percent)
#based on this plot lets only include observations with balances from -1000 to 10,000
exclude <- c(-8020:-1001, 10001:102128)

df <- df %>%
  filter(!(df$balance %in% exclude))
summary(df$balance)
as.data.frame(table(df$y))
#yes responses are still 13.08% which is still good 

#CHANGE FROM CHR TO FACTOR 
df$job = as.factor(df$job)
df$marital = as.factor(df$marital)
df$education = as.factor(df$education)
df$default = as.factor(df$default)
df$housing = as.factor(df$housing)
df$loan = as.factor(df$loan)
df$y = as.factor(df$y)

#----------------------------------------------------------------------------------

## FEATURE ENGINEERING - ADDING YEAR & DATE VARIABLE 
#adding addtional variables 
df <- df %>% mutate(sequence = seq_along(month))

# Vector for storing year
years <- rep(2008, nrow(df))

# Iterating through  "month" column - assign year
for (i in 1:(nrow(df) - 1)) {
  month <- df$month[i]
  if (month == "dec" && df$month[i + 1] == "jan") {
    years[(i + 1):nrow(df)] <- years[i] + 1
  }
}

# Adding "year" column - maintaining the order
df <- df %>% mutate(year = years) %>%
  arrange(sequence) %>%
  select(-sequence)

# For 2008-2010 dataset , creating Date using month and day and year we assigned 
df$Date <- as.Date(paste(df$year, df$month, df$day, sep = "-"), format = "%Y-%b-%d")

# TIME SERIES PLOT  
#plot of entire data 
ggplot(df, aes(x =Date)) + 
  geom_line(stat = "count", aes(color = y), size = 1) +
  theme_minimal() + scale_x_date(date_labels = "%b-%Y", date_breaks  ="1 month") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
  xlab("Month") +  ylab("Count") + ggtitle("Successful vs Nonsucessful Clients by Month") 

# some trends: 
# we can see that there is a peak from Nov2008 to Dec2008
# we can also see that there are greater sucesses from Mid April to Mid June 2009
# could be that we saw more activity during the summer months
# through 2008 to 2010, its interesting how the no responses tend to mirror the yes responses
# this could be because there are more no and less yes responses 
#-------------------------------------------------------------------------------------
#CHECKING FOR BIAS/IMBALANCE
# The dataset is already biased in that there are a lot of unscessful responses compared to sucesses
# checking target imbalance 
tgt_imb <- table(df$y)
tgt_imb

#SPLIT DATA AND TREATING IMBALANCE USING ROSE 
# Random seeding so every run gives consistent data set 
# set.seed(123)
# 
# # Splitting the data into train and test sets
# train_indices <- createDataPartition(df$y, p = 0.7, list = FALSE)
# train_data <- df[train_indices, ]
# test_data <- df[-train_indices, ]

set.seed(123)
index<-createDataPartition(df$y,p=0.7,list=FALSE)
train<-df[index,]
test<-df[-index,]

#count of responses for yes and no are imbalanced for train data 
train_tgt_imb <- table(train$y)
print(train_tgt_imb)

test_tgt_imb <- table(test$y)
print(test_tgt_imb)

#undersampled-sample the majority class with ROSE
train_balsam <- ovun.sample(y ~ ., data = train, seed = 123, method = "both")$data
table(train_balsam$y)
#responses are now balanced 

#---------------------------------------------------------------------------------------

#SVM MODEL 

#note: this code for tuning is commented out as it takes hours to run, however results are shown below after based on the tune model below
#tuning the model to find the best parameters for cost and gamma
# start_time <- Sys.time()
# tune_out=tune(svm, y ~ . -year -Date -day -month ,data=train_balsam,
#               type = "C-classification",
#               kernel = "radial",
#               ranges = list( cost = c(0.1,1,10,100) , gamma = c(0.001,0.01,0.1,0.90)))
# end_time <- Sys.time()
# time.elapse <- (end_time - start_time)
# print(time.elapse)
# summary(tune_out)

#RESULTS FOR TUNING
# Time difference of 8.44494 hours
# Parameter tuning of ‘svm’:
#   
#   - sampling method: 10-fold cross validation 
# 
# - best parameters:
#   cost gamma
# 100   0.9
# 
# - best performance: 0.1379568 
# 
# - Detailed performance results:
#   cost gamma     error  dispersion
# 1    0.1 0.001 0.3580387 0.010433610
# 2    1.0 0.001 0.3486095 0.009625188
# 3   10.0 0.001 0.3435739 0.009243624
# 4  100.0 0.001 0.3355644 0.009538138
# 5    0.1 0.010 0.3398900 0.009520132
# 6    1.0 0.010 0.3347869 0.009203579
# 7   10.0 0.010 0.3322521 0.009979144
# 8  100.0 0.010 0.3256955 0.010649896
# 9    0.1 0.100 0.3302920 0.010988568
# 10   1.0 0.100 0.3178548 0.009196000
# 11  10.0 0.100 0.2963942 0.009903945
# 12 100.0 0.100 0.2657749 0.007078421
# 13   0.1 0.900 0.2844979 0.009830379
# 14   1.0 0.900 0.2163310 0.010199858
# 15  10.0 0.900 0.1685767 0.011194470
# 16 100.0 0.900 0.1379568 0.007843234

#We can see that the best paramaters are cost =100 gamma = 0.90

#running the model with best parameters for cost=100 and gama=0.9
start_time <- Sys.time()
model_SVM = svm(formula = y ~. -year -Date -day -month,
                data = train_balsam,
                type = 'C-classification', # this is because we want to make a regression classification
                kernel = 'radial',
                cost = 100,
                gamma = 0.9)
end_time <- Sys.time()
time.elapse <- (end_time - start_time)
print(time.elapse)
#Time difference of 6.900817 mins
summary(model_SVM)
# Call:
#   svm(formula = y ~ . - year - Date - day - month - month_year, data = train_balsam, 
#       type = "C-classification", kernel = "radial", cost = 100, gamma = 0.9)
# 
# 
# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
# cost:  100 
# 
# Number of Support Vectors:  14250
# 
# ( 5998 8252 )
# 
# 
# Number of Classes:  2 
# 
# Levels: 
#   no yes

Predicted_value = predict(model_SVM, test, type="class")
#create confusion matrix 
cm <- confusionMatrix(Predicted_value, reference = test$y, positive = "yes")
#plot confusion matrix 
plt <- as.data.frame(cm$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") 

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   no  yes
# no  8636  891
# yes 2577  576
# 
# Accuracy : 0.7265          
# 95% CI : (0.7186, 0.7342)
# No Information Rate : 0.8843          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.1086          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.39264         
#             Specificity : 0.77018         
#          Pos Pred Value : 0.18268         
#          Neg Pred Value : 0.90648         
#              Prevalence : 0.11569         
#          Detection Rate : 0.04543         
#    Detection Prevalence : 0.24866         
#       Balanced Accuracy : 0.58141         
#                                           
#        'Positive' Class : yes                            


##Confirming Stats 
TP=576
FP=2577
TN=8636
FN=891
precision = TP/(TP+FP)
precision
sensitivity = TP/(TP+FN)#also known as recall
specificity = TN/(TN+FP)

#sensitivity measures the proportion of actual churners which are correctly identified as such and specificity corresponds to the proportion of non-churners which are correctly identified.

##INTERPRETATION BASED ON FINAL SVM MODEL 
# The model will catch 39.3% of clients who will actually suscribe to a term deposit.sensitivity 
# The model will catch 77.0% of clients who will actually not suscribe to a term deposit.specificity
# Overall all accuracy is 72.3%. Error rate is 27.7%.-> Decent model but could be better.
# Out of the clients it predicted as will suscribe, 18.3% of them will actually suscribe. pos pred value
# Out of the customers it predicted as will not suscribe, 90.6% of them will actually not suscribe. neg pred value 
# Computationally expensive as number of observations increase so may not be the best model to use.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#RESULTS BEFORE TUNING (ADDITIONAL INSIGHT)
# Call:
#   svm(formula = y ~ . - year - Date - day - month - month_year, data = train_balsam, 
#       type = "C-classification", kernel = "radial")
# 
# 
# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
# cost:  1 
# 
# Number of Support Vectors:  21907
# 
# ( 10958 10949 )
# 
# 
# Number of Classes:  2 
# 
# Levels: 
#   no yes

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   no  yes
# no  8600  658
# yes 2851  848
# 
# Accuracy : 0.7292          
# 95% CI : (0.7214, 0.7368)
# No Information Rate : 0.8838          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.1924          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.56308         
#             Specificity : 0.75103         
#          Pos Pred Value : 0.22925         
#          Neg Pred Value : 0.92893         
#              Prevalence : 0.11623         
#          Detection Rate : 0.06545         
#    Detection Prevalence : 0.28548         
#       Balanced Accuracy : 0.65705         
#                                           
#        'Positive' Class : yes                     