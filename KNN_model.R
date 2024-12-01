rm(list = ls())

df <- read.csv("C:/Users/sneha/OneDrive/Documents/Divya OMSA/OMSA 2023 - 2024/Summer 2023/MGT 6203 Data Analytics in Business/Project/bank+marketing/bank/bank-full.csv", sep=';')

library(dplyr)
library(tidyverse)
library(scales)
library(caret)
library(ROSE)
library(e1071)
library(fastDummies)
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
boxplot(df$balance, df$campaign, df$previous)
par(mfrow=c(2,2))
boxplot(df$balance, xlab = "Balance")
boxplot(df$campaign, xlab = "Campaign")
boxplot(df$previous, xlab = "Previous")
# it seems like there are a lot of outliers for each numeric attribute, especially Balance
# however we cannot say for sure they are outliers without other conext 
summary(df$balance)
ggplot(df,aes(x = df$balance)) + geom_histogram(aes(y = after_stat(count / sum(count))), color="black", fill="light blue", binwidth = 100) + 
  labs(title = "Balance Amount Distribution") + xlim(-1000,10000) +
  stat_bin(
    binwidth = 100, geom = "text", color = "black",
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

head(df)

## ADDING YEAR & DATE VARIABLE 
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

# DON'T THINK ITS NECESSARY TO INCLUDE WEEKDAYS
# Adding Weekday as a new attribute to the df
# df$Weekday <- weekdays(df$Date)


# TIME SERIES PLOT  
#plot of entire data 
ggplot(df, aes(x =Date)) + 
  geom_line(stat = "count", aes(color = y), size = 1) +
  theme_minimal() + scale_x_date(date_labels = "%b-%Y", date_breaks  ="1 month") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
  xlab("Month") +  ylab("Count") + ggtitle("Successful vs Nonsucessful Clients by Month") 

#same plot during different time intervals for further analysis 
start = "2008-11-01"
end = "2008-12-31"

ggplot(df, aes(x =Date)) + 
  geom_line(stat = "count", aes(color = y), size = 1) +
  theme_minimal() + scale_x_date(date_labels = "%b-%Y", date_breaks  ="1 month", limits = as.Date(c(start, end))) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
  xlab("Month") +  ylab("Count") + ggtitle("Successful vs Nonsucessful Clients by Month") 
# we can see that there is a peak from Nov2008 to Dec2008
# we can also see that there are greater sucesses from Mid April to Mid June 2009
# could be that we saw more activity during the summer months
# through 2008 to 2010, its interesting how the no responses tend to mirror the yes responses
# this could be because there are more no and less yes responses 

#join date and year 
df$year <- as.character(df$year)
df$month_year<- with(df, paste0(month,year))
#now changing month_year to as factor 
df$month_year = as.factor(df$month_year)

#-------------------------------------------------------------------------------------

# #CREATION OF DUMMY VARS
# df$month = as.factor(df$month)
# dummy_data <- fastDummies::dummy_cols(df[,c(1:13)],remove_first_dummy = FALSE)
# str(dummy_data)
# 
# df1 <- dummy_data[ ,  !names(dummy_data) %in% 
#       c("job", "marital", "education", "default", "housing","loan","month")]
# 
# df <- cbind(df1,y = df$y )
# str(df)

#-----------------------------------------------------------------------------------

#CHECKING FOR BIAS/IMBALANCE
# The dataset is already biased in that there are a lot of unscessful responses compared to sucesses
# checking target imbalance 
tgt_imb <- table(df$y)
tgt_imb

head(df)

#SPLIT DATA AND TREATING IMBALANCE USING ROSE 
# Random seeding so every run gives consistent data set 
# set.seed(123)
# 
# # Splitting the data into train and test sets
# train_indices <- createDataPartition(df$y, p = 0.7, list = FALSE)
# train_data <- df[train_indices, ]
# test_data <- df[-train_indices, ]

# Split the train_balsam dataset into training features, training target, testing features, and testing target

#-------------------------------------------------------------------------------------------
# MODEL 1 - KNN without balancing the data, and setting k as 5 & 100

library(class)

# Removing the unwanted columns
df <- df[, !(names(df) %in% c("year", "Date", "day", "month", "month_year"))]
# Checking if the columns are removed successfully
head(df) 

# Convert binary variables to 0 and 1
df$y <- ifelse(df$y == "yes", 1, 0)
df$default <- ifelse(df$default == "yes", 1, 0)
df$housing <- ifelse(df$housing == "yes", 1, 0)
df$loan <- ifelse(df$loan == "yes", 1, 0)

# Convert categorical variables to numeric factors
df$job <- as.numeric(factor(df$job))
df$marital <- as.numeric(factor(df$marital))
df$education <- as.numeric(factor(df$education))

head(df)

# Splitting the data into train and test sets
set.seed(123)
index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[index,]
test <- df[-index,]

# Separating train features and target variable
train_features <- train[, !(names(train) %in% c("y"))]
train_target <- train$y

# Removing unwanted columns from test features
test_features <- test[, !(names(test) %in% c("year", "Date", "day", "month", "month_year", "y"))]
test_target <- test$y

# Let us scale the age and balance columns inorder to run Knn
scaled_train_features <- train
scaled_train_features[, c("age", "balance")] <- scale(train[, c("age", "balance")])

scaled_test_features <- test
scaled_test_features[, c("age", "balance")] <- scale(test[, c("age", "balance")])

# Lets train the KNN model with k=5 
model_KNN <- knn(train = scaled_train_features, test = scaled_test_features, cl = train_target, k = 5)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

# Now with k=100 
model_KNN <- knn(train = scaled_train_features, test = scaled_test_features, cl = train_target, k = 100)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

#---------------------------------------------------------------------------------------

# MODEL 2 - KNN after balancing the data, working with different k values

#Lets count no. of responses for yes and no are imbalanced for train data 
train_tgt_imb <- table(train$y)
print(train_tgt_imb)
test_tgt_imb <- table(test$y)
print(test_tgt_imb)

# Undersampling the majority cass
train_balsam <- ovun.sample(y ~ ., data = train, seed = 123, method = "both")$data
table(train_balsam$y)

train_features <- train_balsam[, !(names(train_balsam) %in% c("year", "Date", "day", "month", "month_year", "y"))]
train_target <- train_balsam$y
#head(train_target)
test_features <- test[, !(names(test) %in% c("year", "Date", "day", "month", "month_year", "y"))]
test_target <- test$y

# scaling the data before performing knn
variables_to_scale <- c("age", "balance")
train_features[variables_to_scale] <- scale(train_features[variables_to_scale])

# Checking the scaled dataset
head(train_features)

variables_to_scale_test <- c("age", "balance")
test_features[variables_to_scale_test] <- scale(test_features[variables_to_scale_test])
# Check the scaled dataset
head(test_features)

# Lets train the KNN model with k value as 5 and 100
model_KNN <- knn(train = train_features, test = test_features, cl = train_target, k = 5)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf


model_KNN <- knn(train = train_features, test = test_features, cl = train_target, k = 100)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

#-----------------------------------------------------------------------------------

# MODEL 3 - Using Cross validation to find an optimal k value and performing the KNN

# Checking optimal k values using cross validation technique for processed and balanced data

# Optimal K value for processed data
#set.seed(123)
#control <- trainControl(method = "cv", number = 5)

# we have to convert the variable to a factor
#df$y <- factor(df$y, levels = c(0, 1))

# Training the KNN model using cross-validation
#model <- train(y ~ ., data = df, method = "knn", trControl = control, tuneLength = 10)
#print(model$results)
#best_k <- model$bestTune$k
#best_k

# Finding the best K for balanced datset

#train_features <- train_balsam[, !(names(train_balsam) %in% c("year", "Date", "day", "month", "month_year"))]
#train_target <- train_balsam$y
#test_features <- test[, !(names(test) %in% c("year", "Date", "day", "month", "month_year"))]
#test_target <- test$y

#train_balsam$y <- factor(train_balsam$y, levels = c(0, 1))
#model <- train(y ~ ., data = train_balsam, method = "knn", trControl = control, tuneLength = 50)
#print(model$results)
#best_k <- model$bestTune$k
#best_k

model_KNN <- knn(train = scaled_train_features, test = scaled_test_features, cl = train_target, k = 18)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

model_KNN <- knn(train = train_features, test = test_features, cl = train_target, k = 5)
pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

#------------------------------------------------------------------------------------------

# MODEL 4 - Performing One hot encoding and doing knn

# Doing One-Hot Encoding
df <- df %>%
  mutate(job = as.character(job),
         marital = as.character(marital),
         education = as.character(education)) %>%
  dummy_cols(select_columns = c("job", "marital", "education"), remove_first_dummy = FALSE)
df <- df %>%
  select(-c(job, marital, education))


head(df)

tgt_imb <- table(df$y)
tgt_imb

df$default <- ifelse(df$default == "yes", 1, 0)
df$housing <- ifelse(df$housing == "yes", 1, 0)
df$loan <- ifelse(df$loan == "yes", 1, 0)

head(df)

set.seed(123)
control <- trainControl(method = "cv", number = 5)

#Converting outcome variable to factor with two levels
df$y <- factor(df$y, levels = c(0, 1))

#Training the KNN model using cross-validation
model <- train(y ~ ., data = df, method = "knn", trControl = control, tuneLength = 10)
print(model$results)
best_k <- model$bestTune$k
best_k

set.seed(123)
index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[index,]
test <- df[-index,]

train_tgt_imb <- table(train$y)
print(train_tgt_imb)

test_tgt_imb <- table(test$y)
print(test_tgt_imb)

colnames(train)[colnames(train) == 'job_blue-collar'] <- 'job_blue_collar'
colnames(test)[colnames(test) == 'job_blue-collar'] <- 'job_blue_collar'
colnames(train)[colnames(train) == 'job_self-employed'] <- 'job_self_employed'
colnames(test)[colnames(test) == 'job_self-employed'] <- 'job_self_employed'

train_balsam <- ovun.sample(y ~ ., data = train, seed = 123, method = "both")$data
table(train_balsam$y)

train_features <- train_balsam[, !(names(train_balsam) %in% c("year", "Date", "day", "month", "month_year", "y"))]
train_target <- train_balsam$y
#head(train_target)

test_features <- test[, !(names(test) %in% c("year", "Date", "day", "month", "month_year", "y"))]
test_target <- test$y

#scaling the data
variables_to_scale <- c("age", "balance")
train_features_scaled <- as.data.frame(scale(train_features[variables_to_scale]))
train_features_scaled <- cbind(train_features_scaled, train_features[!names(train_features) %in% variables_to_scale])

# Checking the scaled dataset
head(train_features_scaled)

variables_to_scale_test <- c("age", "balance")

# Scale the selected variables
test_features_scaled <- as.data.frame(scale(test_features[variables_to_scale]))
test_features_scaled <- cbind(test_features_scaled, test_features[!names(test_features) %in% variables_to_scale])

# Check the scaled dataset
head(test_features_scaled)

# Train the KNN model with k = 23 & 100
model_KNN <- knn(train = train_features_scaled, test = test_features_scaled, cl = train_target, k = 23)

pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf

model_KNN <- knn(train = train_features_scaled, test = test_features_scaled, cl = train_target, k = 100)

pred_conf <- confusionMatrix(as.factor(model_KNN), as.factor(test_target), positive = "1")
pred_conf



