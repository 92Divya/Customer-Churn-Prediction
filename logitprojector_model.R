library(MASS)
library(caret)
library(dplyr)
library(fastDummies)
library(ROSE)

set.seed(123)

########## BEGIN DATA PREPARATION ##########

data <- read.csv("C:\\Users\\mholl\\OneDrive\\Documents\\GT Local\\MGT 6203\\bank-full.csv", sep=';')
df <- subset(data, select = -c(duration))

# Creating date column
df <- df %>% mutate(sequence = seq_along(month))
years <- rep(2008, nrow(df))

for (i in 1:(nrow(df) - 1)) {
  month <- df$month[i]
  if (month == "dec" && df$month[i + 1] == "jan") {
    years[(i + 1):nrow(df)] <- years[i] + 1
  }
}

df <- df %>% mutate(year = years) %>%
  arrange(sequence) %>%
  select(-sequence)

df$Date <- as.Date(paste(df$year, df$month, df$day, sep = "-"), format = "%Y-%b-%d")
df$Weekday <- weekdays(df$Date)

# Converting variables to binary where necessary
df$y <- ifelse(df$y == 'no', 0, 1)
df$default <- ifelse(df$default == 'no', 0, 1)
df$housing <- ifelse(df$housing == 'no', 0, 1)
df$loan <- ifelse(df$loan == 'no', 0, 1)

# Replacing "-" with "_" in column names
new_colnames <- gsub("-", "_", colnames(df))
colnames(df) <- new_colnames

# Creating manual dummy variables
char_cols <- sapply(df, is.character)
df_dummies <- dummy_cols(df[, char_cols], remove_first_dummy = FALSE, remove_selected_columns = TRUE)
df <- df[, !char_cols]
df <- cbind(df, df_dummies)

# Splitting the data into train and test sets
train_indices <- createDataPartition(df$y, p = 0.7, list = FALSE)
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

########## BEGIN MODELLING ##########

### Logistic Predictor Model #1: Forward Stepwise Regression from full dataset:

null_model <- glm(y~1, train_data, family=binomial)
full_model <- glm(y~., train_data, family=binomial)

model_f <- stepAIC(null_model, direction = 'forward', scope = list(upper = full_model, lower = null_model))
pred_f_r <- predict(model_f, test_data, type='response')

mf_results <- data.frame(pred_f_r, test_data$y)
mf_results$thresh_50 <- ifelse(mf_results$pred_f_r > 0.49, 1, 0)
mf_results$test_data.y <- as.factor(mf_results$test_data.y)
mf_results$thresh_50 <- as.factor(mf_results$thresh_50)

mf_mat <- confusionMatrix(mf_results$thresh_50, reference = mf_results$test_data.y, positive='1')
mf_mcfad <- with(summary(model_f), 1 - deviance/null.deviance)

# Model #1 Output:

# Call:
#   glm(formula = y ~ Date + poutcome_success + housing + month_mar + 
#         poutcome_unknown + campaign + month_jan + month_may + marital_married + 
#         Weekday_Monday + loan + contact_cellular + month_aug + month_nov + 
#         education_tertiary + job_retired + day + month_jun + contact_telephone + 
#         education_secondary + balance + Weekday_Saturday + job_housemaid + 
#         pdays, family = binomial, data = train_data)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)         -5.165e+01  1.523e+00 -33.906  < 2e-16 ***
#   Date                 3.450e-03  1.056e-04  32.683  < 2e-16 ***
#   poutcome_success     1.788e+00  8.755e-02  20.421  < 2e-16 ***
#   housing             -3.565e-01  4.555e-02  -7.826 5.02e-15 ***
#   month_mar            9.610e-01  1.237e-01   7.767 8.06e-15 ***
#   poutcome_unknown     3.371e-01  9.146e-02   3.685 0.000229 ***
#   campaign            -6.290e-02  9.867e-03  -6.374 1.84e-10 ***
#   month_jan           -1.040e+00  1.227e-01  -8.476  < 2e-16 ***
#   month_may           -3.446e-01  5.769e-02  -5.973 2.33e-09 ***
#   marital_married     -2.377e-01  4.013e-02  -5.923 3.16e-09 ***
#   Weekday_Monday      -3.153e-01  5.054e-02  -6.239 4.41e-10 ***
#   loan                -3.525e-01  6.442e-02  -5.472 4.46e-08 ***
#   contact_cellular     5.242e-01  7.482e-02   7.006 2.46e-12 ***
#   month_aug           -3.705e-01  6.466e-02  -5.730 1.00e-08 ***
#   month_nov           -4.740e-01  7.594e-02  -6.242 4.33e-10 ***
#   education_tertiary   2.678e-01  6.166e-02   4.343 1.40e-05 ***
#   job_retired          2.546e-01  7.781e-02   3.272 0.001069 ** 
#   day                  8.527e-03  2.471e-03   3.451 0.000559 ***
#   month_jun            3.071e-01  8.035e-02   3.822 0.000132 ***
#   contact_telephone    2.831e-01  1.041e-01   2.720 0.006527 ** 
#   education_secondary  1.380e-01  5.798e-02   2.379 0.017346 *  
#   balance              1.546e-05  5.872e-06   2.633 0.008468 ** 
#   Weekday_Saturday    -1.120e+00  6.705e-01  -1.671 0.094767 .  
# job_housemaid       -2.335e-01  1.317e-01  -1.773 0.076206 .  
# pdays               -4.485e-04  3.147e-04  -1.425 0.154199    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 22889  on 31647  degrees of freedom
# Residual deviance: 18266  on 31623  degrees of freedom
# AIC: 18316
# 
# Number of Fisher Scoring iterations: 6
# 
# > mf_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11781  1219
# 1   207   356
# 
# Accuracy : 0.8949       
# 95% CI : (0.8896, 0.9)
# No Information Rate : 0.8839       
# P-Value [Acc > NIR] : 2.75e-05     
# 
# Kappa : 0.2896       
# 
# Mcnemar's Test P-Value : < 2.2e-16    
#                                        
#             Sensitivity : 0.22603      
#             Specificity : 0.98273      
#          Pos Pred Value : 0.63233      
#          Neg Pred Value : 0.90623      
#              Prevalence : 0.11612      
#          Detection Rate : 0.02625      
#    Detection Prevalence : 0.04151      
#       Balanced Accuracy : 0.60438      
#                                        
#        'Positive' Class : 1            
#                                        
# > mf_mcfad
# [1] 0.2019668


### Logistic Predictor Model #2: Backward Stepwise Regression from full dataset:

model_b <- stepAIC(full_model, direction = 'backward', scope = list(upper = full_model, lower = null_model))
pred_b_r <- predict(model_b, test_data, type='response')

mb_results <- data.frame(pred_b_r, test_data$y)
mb_results$thresh_50 <- ifelse(mb_results$pred_b_r > 0.49, 1, 0)
mb_results$test_data.y <- as.factor(mb_results$test_data.y)
mb_results$thresh_50 <- as.factor(mb_results$thresh_50)

mb_mat <- confusionMatrix(mb_results$thresh_50, reference = mb_results$test_data.y, positive='1')
mb_mcfad <- with(summary(model_b), 1 - deviance/null.deviance)

# Model #2 Output:

# Call:
#   glm(formula = y ~ age + default + balance + housing + loan + 
#         day + campaign + pdays + previous + year + Date + job_admin. + 
#         `job_blue-collar` + job_entrepreneur + job_housemaid + job_management + 
#         job_retired + `job_self-employed` + job_services + job_student + 
#         job_technician + job_unemployed + job_unknown + marital_divorced + 
#         marital_married + marital_single + education_primary + education_secondary + 
#         education_tertiary + education_unknown + contact_cellular + 
#         contact_telephone + contact_unknown + month_apr + month_aug + 
#         month_dec + month_feb + month_jan + month_jul + month_jun + 
#         month_mar + month_may + month_nov + month_oct + month_sep + 
#         poutcome_failure + poutcome_other + poutcome_success + poutcome_unknown + 
#         Weekday_Friday + Weekday_Monday, family = binomial, data = train_data)
# 
# Coefficients: (6 not defined because of singularities)
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)          7.519e+11  1.251e+13   0.060 0.952081    
# age                  1.042e-03  2.375e-03   0.439 0.660961    
# default              1.207e-02  1.735e-01   0.070 0.944538    
# balance              1.496e-05  5.915e-06   2.529 0.011426 *  
#   housing             -3.548e-01  4.715e-02  -7.524 5.31e-14 ***
#   loan                -3.390e-01  6.501e-02  -5.215 1.84e-07 ***
#   day                 -1.046e+06  1.741e+07  -0.060 0.952081    
# campaign            -6.119e-02  9.925e-03  -6.165 7.05e-10 ***
#   pdays               -4.133e-04  3.166e-04  -1.305 0.191811    
# previous            -9.085e-03  1.165e-02  -0.780 0.435579    
# year                -3.818e+08  6.354e+09  -0.060 0.952081    
# Date                 1.046e+06  1.741e+07   0.060 0.952081    
# job_admin.           2.971e-01  2.630e-01   1.130 0.258568    
# `job_blue-collar`    2.438e-01  2.619e-01   0.931 0.351837    
# job_entrepreneur     2.659e-01  2.813e-01   0.945 0.344556    
# job_housemaid        6.768e-02  2.858e-01   0.237 0.812790    
# job_management       3.309e-01  2.606e-01   1.270 0.204185    
# job_retired          5.330e-01  2.663e-01   2.002 0.045300 *  
#   `job_self-employed`  3.334e-01  2.757e-01   1.209 0.226596    
# job_services         3.241e-01  2.660e-01   1.218 0.223135    
# job_student          4.406e-01  2.767e-01   1.592 0.111319    
# job_technician       3.172e-01  2.609e-01   1.216 0.224053    
# job_unemployed       3.664e-01  2.775e-01   1.320 0.186736    
# job_unknown                 NA         NA      NA       NA    
# marital_divorced     4.735e-02  7.221e-02   0.656 0.511951    
# marital_married     -2.144e-01  5.007e-02  -4.282 1.85e-05 ***
#   marital_single              NA         NA      NA       NA    
# education_primary   -4.477e-02  1.151e-01  -0.389 0.697286    
# education_secondary  9.465e-02  1.022e-01   0.926 0.354263    
# education_tertiary   2.107e-01  1.071e-01   1.967 0.049153 *  
#   education_unknown           NA         NA      NA       NA    
# contact_cellular     5.459e-01  7.603e-02   7.180 6.96e-13 ***
#   contact_telephone    3.036e-01  1.054e-01   2.881 0.003964 ** 
#   contact_unknown             NA         NA      NA       NA    
# month_apr            1.601e+08  2.663e+09   0.060 0.952081    
# month_aug            3.243e+07  5.396e+08   0.060 0.952081    
# month_dec           -9.520e+07  1.584e+09  -0.060 0.952081    
# month_feb            2.218e+08  3.690e+09   0.060 0.952081    
# month_jan            2.542e+08  4.230e+09   0.060 0.952081    
# month_jul            6.486e+07  1.079e+09   0.060 0.952081    
# month_jun            9.624e+07  1.602e+09   0.060 0.952081    
# month_mar            1.925e+08  3.203e+09   0.060 0.952081    
# month_may            1.287e+08  2.141e+09   0.060 0.952081    
# month_nov           -6.381e+07  1.062e+09  -0.060 0.952081    
# month_oct           -3.138e+07  5.222e+08  -0.060 0.952081    
# month_sep                   NA         NA      NA       NA    
# poutcome_failure    -3.558e-01  1.037e-01  -3.431 0.000602 ***
#   poutcome_other      -2.201e-01  1.230e-01  -1.789 0.073567 .  
# poutcome_success     1.475e+00  1.001e-01  14.733  < 2e-16 ***
#   poutcome_unknown            NA         NA      NA       NA    
# Weekday_Friday      -1.082e-02  5.111e-02  -0.212 0.832325    
# Weekday_Monday      -3.195e-01  5.210e-02  -6.132 8.65e-10 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 22889  on 31647  degrees of freedom
# Residual deviance: 18259  on 31602  degrees of freedom
# AIC: 18351
# 
# Number of Fisher Scoring iterations: 25
# 
# > mb_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11790  1215
# 1   198   360
# 
# Accuracy : 0.8958          
# 95% CI : (0.8906, 0.9009)
# No Information Rate : 0.8839          
# P-Value [Acc > NIR] : 5.608e-06       
# 
# Kappa : 0.2947          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.22857         
#             Specificity : 0.98348         
#          Pos Pred Value : 0.64516         
#          Neg Pred Value : 0.90657         
#              Prevalence : 0.11612         
#          Detection Rate : 0.02654         
#    Detection Prevalence : 0.04114         
#       Balanced Accuracy : 0.60603         
#                                           
#        'Positive' Class : 1               
#                                           
# > mb_mcfad
# [1] 0.2022969

### Logistic Predictor Model #3: Manual Variable selection based on p-values, removing call-timing attributes from the dataset

# Removing call-timing variables
train_manual <- subset(train_data, select = -c(day, year, Date, month_jan, month_feb,
                                               month_mar, month_apr, month_may, month_jun, month_jul,
                                               month_aug, month_sep, month_oct, month_nov, month_dec,
                                               Weekday_Monday, Weekday_Tuesday, 
                                               Weekday_Wednesday, Weekday_Thursday, Weekday_Friday, 
                                               Weekday_Saturday, Weekday_Sunday))

test_manual <- subset(test_data, select = -c(day, year, Date, month_jan, month_feb,
                                             month_mar, month_apr, month_may, month_jun, month_jul,
                                             month_aug, month_sep, month_oct, month_nov, month_dec,
                                             Weekday_Monday, Weekday_Tuesday, 
                                             Weekday_Wednesday, Weekday_Thursday, Weekday_Friday, 
                                             Weekday_Saturday, Weekday_Sunday))

# Building baseline model with all remaining attributes
model_m1 <- glm(y~., train_manual, family=binomial)
pred_m1 <- predict(model_m1, test_manual, type='response')

results_m1 <- data.frame(pred_m1, test_manual$y)
results_m1$thresh_50 <- ifelse(results_m1$pred_m1 > 0.49, 1, 0)
results_m1$test_manual.y <- as.factor(results_m1$test_manual.y)
results_m1$thresh_50 <- as.factor(results_m1$thresh_50)

m1_mat <- confusionMatrix(results_m1$thresh_50, reference = results_m1$test_manual.y, positive='1')
m1_mcfad <- with(summary(model_m1), 1 - deviance/null.deviance)
m1_roc <- roc.curve(results_m1$test_manual.y, results_m1$thresh_50)

# Creating second model by removing factors w/ non-significant p-values:

train_manual2 <- subset(train_manual, select = c(y, balance, housing, loan, campaign, pdays, job_retired,
                                                 job_student, marital_married, contact_cellular,
                                                 contact_telephone, poutcome_other, poutcome_success))

test_manual2 <- subset(test_manual, select = c(y, balance, housing, loan, campaign, pdays, job_retired,
                                               job_student, marital_married, contact_cellular,
                                               contact_telephone, poutcome_other, poutcome_success))

model_m2 <- glm(y~., train_manual2, family=binomial)
pred_m2 <- predict(model_m2, test_manual2, type='response')

results_m2 <- data.frame(pred_m2, test_manual2$y)
results_m2$thresh_50 <- ifelse(results_m2$pred_m2 > 0.49, 1, 0)
results_m2$test_manual2.y <- as.factor(results_m2$test_manual2.y)
results_m2$thresh_50 <- as.factor(results_m2$thresh_50)

m2_mat <- confusionMatrix(results_m2$thresh_50, reference = results_m2$test_manual2.y, positive='1')
m2_mcfad <- with(summary(model_m2), 1 - deviance/null.deviance)
m2_roc <- roc.curve(results_m2$test_manual2.y, results_m2$thresh_50)

# Model #3 Output:

# Call:
#   glm(formula = y ~ ., family = binomial, data = train_manual2)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)       -2.297e+00  6.643e-02 -34.568  < 2e-16 ***
#   balance            2.414e-05  5.353e-06   4.509 6.51e-06 ***
#   housing           -6.036e-01  4.052e-02 -14.896  < 2e-16 ***
#   loan              -5.482e-01  6.232e-02  -8.796  < 2e-16 ***
#   campaign          -1.098e-01  1.012e-02 -10.847  < 2e-16 ***
#   pdays              5.703e-04  1.961e-04   2.909  0.00363 ** 
#   job_retired        5.706e-01  7.077e-02   8.063 7.46e-16 ***
#   job_student        5.996e-01  1.002e-01   5.982 2.20e-09 ***
#   marital_married   -3.173e-01  3.878e-02  -8.180 2.83e-16 ***
#   contact_cellular   1.007e+00  5.853e-02  17.204  < 2e-16 ***
#   contact_telephone  8.444e-01  8.965e-02   9.419  < 2e-16 ***
#   poutcome_other     2.462e-01  8.902e-02   2.766  0.00568 ** 
#   poutcome_success   2.315e+00  7.398e-02  31.286  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 22889  on 31647  degrees of freedom
# Residual deviance: 19773  on 31635  degrees of freedom
# AIC: 19799
# 
# Number of Fisher Scoring iterations: 6
# 
# > m2_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11841  1297
# 1   147   278
# 
# Accuracy : 0.8935          
# 95% CI : (0.8882, 0.8987)
# No Information Rate : 0.8839          
# P-Value [Acc > NIR] : 0.0002018       
# 
# Kappa : 0.2405          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.17651         
#             Specificity : 0.98774         
#          Pos Pred Value : 0.65412         
#          Neg Pred Value : 0.90128         
#              Prevalence : 0.11612         
#          Detection Rate : 0.02050         
#    Detection Prevalence : 0.03134         
#       Balanced Accuracy : 0.58212         
#                                           
#        'Positive' Class : 1               
#                                           
# > m2_mcfad
# [1] 0.1361404


### Logistic Predictor Model #4: Forward Stepwise Regression removing call-timing attributes from the dataset:

null_model2 <- glm(y~1, train_manual, family=binomial)
full_model2 <- glm(y~., train_manual, family=binomial)

model_f2 <- stepAIC(null_model2, direction = 'forward', scope = list(upper = full_model2, lower = null_model2))
pred_f2_r <- predict(model_f2, test_data, type='response')

mf2_results <- data.frame(pred_f2_r, test_manual$y)
mf2_results$thresh_50 <- ifelse(mf2_results$pred_f2_r > 0.49, 1, 0)
mf2_results$test_manual.y <- as.factor(mf2_results$test_manual.y)
mf2_results$thresh_50 <- as.factor(mf2_results$thresh_50)

mf2_mat <- confusionMatrix(mf2_results$thresh_50, reference = mf2_results$test_manual.y, positive='1')
mf2_mcfad <- with(summary(model_f2), 1 - deviance/null.deviance)

# Model #4 Output:

# Call:
#   glm(formula = y ~ poutcome_success + contact_unknown + housing + 
#         campaign + loan + marital_married + job_retired + job_student + 
#         education_tertiary + pdays + balance + education_primary + 
#         job_admin. + poutcome_other + job_unemployed + contact_cellular + 
#         job_management, family = binomial, data = train_manual)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)        -1.553e+00  8.704e-02 -17.838  < 2e-16 ***
#   poutcome_success    2.289e+00  7.410e-02  30.894  < 2e-16 ***
#   contact_unknown    -8.373e-01  8.979e-02  -9.325  < 2e-16 ***
#   housing            -5.821e-01  4.074e-02 -14.288  < 2e-16 ***
#   campaign           -1.093e-01  1.012e-02 -10.798  < 2e-16 ***
#   loan               -5.340e-01  6.250e-02  -8.544  < 2e-16 ***
#   marital_married    -2.816e-01  3.917e-02  -7.190 6.49e-13 ***
#   job_retired         6.967e-01  7.380e-02   9.439  < 2e-16 ***
#   job_student         6.960e-01  1.021e-01   6.815 9.41e-12 ***
#   education_tertiary  1.728e-01  5.037e-02   3.431 0.000601 ***
#   pdays               6.012e-04  1.962e-04   3.065 0.002179 ** 
#   balance             2.173e-05  5.379e-06   4.039 5.37e-05 ***
#   education_primary  -1.706e-01  6.286e-02  -2.715 0.006634 ** 
#   job_admin.          1.947e-01  6.165e-02   3.158 0.001589 ** 
#   poutcome_other      2.416e-01  8.901e-02   2.714 0.006638 ** 
#   job_unemployed      2.431e-01  1.055e-01   2.303 0.021264 *  
#   contact_cellular    1.277e-01  7.426e-02   1.720 0.085497 .  
# job_management      8.097e-02  5.678e-02   1.426 0.153849    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 22889  on 31647  degrees of freedom
# Residual deviance: 19721  on 31630  degrees of freedom
# AIC: 19757
# 
# Number of Fisher Scoring iterations: 6
# 
# > mf2_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11844  1301
# 1   144   274
# 
# Accuracy : 0.8935          
# 95% CI : (0.8881, 0.8986)
# No Information Rate : 0.8839          
# P-Value [Acc > NIR] : 0.0002239       
# 
# Kappa : 0.2378          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.17397         
#             Specificity : 0.98799         
#          Pos Pred Value : 0.65550         
#          Neg Pred Value : 0.90103         
#              Prevalence : 0.11612         
#          Detection Rate : 0.02020         
#    Detection Prevalence : 0.03082         
#       Balanced Accuracy : 0.58098         
#                                           
#        'Positive' Class : 1               
#                                           
# > mf2_mcfad
# [1] 0.1384175

### Logistic Predictor Model #5: Backward Stepwise Regression removing call-timing attributes from the dataset:

model_b2 <- stepAIC(full_model2, direction = 'backward', scope = list(upper = full_model2, lower = null_model2))
pred_b2_r <- predict(model_b2, test_data, type='response')

mb2_results <- data.frame(pred_b2_r, test_manual$y)
mb2_results$thresh_50 <- ifelse(mb2_results$pred_b2_r > 0.49, 1, 0)
mb2_results$test_manual.y <- as.factor(mb2_results$test_manual.y)
mb2_results$thresh_50 <- as.factor(mb2_results$thresh_50)

mb2_mat <- confusionMatrix(mb2_results$thresh_50, reference = mb2_results$test_manual.y, positive='1')
mb2_mcfad <- with(summary(model_b2), 1 - deviance/null.deviance)

# Model #5 Output:

# Call:
#   glm(formula = y ~ balance + housing + loan + campaign + pdays + 
#         job_admin. + job_management + job_retired + job_student + 
#         job_unemployed + marital_married + education_primary + education_tertiary + 
#         contact_cellular + contact_telephone + poutcome_other + poutcome_success, 
#       family = binomial, data = train_manual)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)        -2.390e+00  7.063e-02 -33.836  < 2e-16 ***
#   balance             2.173e-05  5.379e-06   4.039 5.37e-05 ***
#   housing            -5.821e-01  4.074e-02 -14.288  < 2e-16 ***
#   loan               -5.340e-01  6.250e-02  -8.544  < 2e-16 ***
#   campaign           -1.093e-01  1.012e-02 -10.798  < 2e-16 ***
#   pdays               6.012e-04  1.962e-04   3.065 0.002179 ** 
#   job_admin.          1.947e-01  6.165e-02   3.158 0.001589 ** 
#   job_management      8.097e-02  5.678e-02   1.426 0.153849    
# job_retired         6.967e-01  7.380e-02   9.439  < 2e-16 ***
#   job_student         6.960e-01  1.021e-01   6.815 9.41e-12 ***
#   job_unemployed      2.431e-01  1.055e-01   2.303 0.021264 *  
#   marital_married    -2.816e-01  3.917e-02  -7.190 6.49e-13 ***
#   education_primary  -1.706e-01  6.286e-02  -2.715 0.006634 ** 
#   education_tertiary  1.728e-01  5.037e-02   3.431 0.000601 ***
#   contact_cellular    9.650e-01  5.890e-02  16.384  < 2e-16 ***
#   contact_telephone   8.373e-01  8.979e-02   9.325  < 2e-16 ***
#   poutcome_other      2.416e-01  8.901e-02   2.714 0.006638 ** 
#   poutcome_success    2.289e+00  7.410e-02  30.894  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 22889  on 31647  degrees of freedom
# Residual deviance: 19721  on 31630  degrees of freedom
# AIC: 19757
# 
# Number of Fisher Scoring iterations: 6
# 
# > mb2_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11844  1301
# 1   144   274
# 
# Accuracy : 0.8935          
# 95% CI : (0.8881, 0.8986)
# No Information Rate : 0.8839          
# P-Value [Acc > NIR] : 0.0002239       
# 
# Kappa : 0.2378          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.17397         
#             Specificity : 0.98799         
#          Pos Pred Value : 0.65550         
#          Neg Pred Value : 0.90103         
#              Prevalence : 0.11612         
#          Detection Rate : 0.02020         
#    Detection Prevalence : 0.03082         
#       Balanced Accuracy : 0.58098         
#                                           
#        'Positive' Class : 1               
#                                           
# > mb2_mcfad
# [1] 0.1384175

### Logistic Predictor Model #6: Manual Variable selection based on p-values, removing call-timing attributes from the dataset, using balanced data

# Creating synthetic data
new_colnames <- gsub("-", "_", colnames(train_manual))
colnames(train_manual) <- new_colnames
new_colnames <- gsub("-", "_", colnames(test_manual))
colnames(test_manual) <- new_colnames

train_manual_ovun <- ovun.sample(y~., train_manual, method='both', seed=1)

# Creating baseline model with all remaining attributes
model_m1_ovun <- glm(y~., train_manual_ovun$data, family=binomial)
pred_m1_ovun <- predict(model_m1_ovun, test_manual, type='response')

results_m1_ovun <- data.frame(pred_m1_ovun, test_manual$y)
results_m1_ovun$thresh_50 <- ifelse(results_m1_ovun$pred_m1_ovun > 0.49, 1, 0)
results_m1_ovun$test_manual.y <- as.factor(results_m1_ovun$test_manual.y)
results_m1_ovun$thresh_50 <- as.factor(results_m1_ovun$thresh_50)

m1_mat_ovun <- confusionMatrix(results_m1_ovun$thresh_50, reference = results_m1_ovun$test_manual.y, positive='1')
m1_mcfad_ovun <- with(summary(model_m1_ovun), 1 - deviance/null.deviance)
m1_roc_ovun <- roc.curve(results_m1_ovun$test_manual.y, results_m1_ovun$thresh_50)

# Creating second model by removing factors w/ non-significant p-values:

train_manual3 <- subset(train_manual_ovun$data, select = c(y, balance, housing, loan, campaign, job_retired,
                                                           job_student, marital_married, marital_divorced, education_tertiary,
                                                           contact_cellular, contact_telephone, poutcome_other, poutcome_success))

test_manual3 <- subset(test_manual, select = c(y, balance, housing, loan, campaign, job_retired,
                                               job_student, marital_married, marital_divorced, education_tertiary,
                                               contact_cellular, contact_telephone, poutcome_other, poutcome_success))

model_m2_ovun <- glm(y~., train_manual3, family=binomial)
pred_m2_ovun <- predict(model_m2_ovun, test_manual3, type='response')

results_m2_ovun <- data.frame(pred_m2_ovun, test_manual3$y)
results_m2_ovun$thresh_50 <- ifelse(results_m2_ovun$pred_m2_ovun > 0.49, 1, 0)
results_m2_ovun$test_manual3.y <- as.factor(results_m2_ovun$test_manual3.y)
results_m2_ovun$thresh_50 <- as.factor(results_m2_ovun$thresh_50)

m2_mat_ovun <- confusionMatrix(results_m2_ovun$thresh_50, reference = results_m2_ovun$test_manual3.y, positive='1')
m2_mcfad_ovun <- with(summary(model_m2_ovun), 1 - deviance/null.deviance)
m2_roc_ovun <- roc.curve(results_m2_ovun$test_manual3.y, results_m2_ovun$thresh_50)

# Model #6 Output:

# Call:
#   glm(formula = y ~ ., family = binomial, data = train_manual3)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)        -3.211e-01  4.274e-02  -7.514 5.72e-14 ***
#   balance             2.500e-05  4.267e-06   5.859 4.66e-09 ***
#   housing            -5.298e-01  2.597e-02 -20.402  < 2e-16 ***
#   loan               -4.162e-01  3.796e-02 -10.965  < 2e-16 ***
#   campaign           -9.487e-02  5.579e-03 -17.004  < 2e-16 ***
#   job_retired         6.988e-01  5.420e-02  12.893  < 2e-16 ***
#   job_student         6.818e-01  8.137e-02   8.379  < 2e-16 ***
#   marital_married    -4.307e-01  2.871e-02 -14.999  < 2e-16 ***
#   marital_divorced   -1.907e-01  4.300e-02  -4.434 9.27e-06 ***
#   education_tertiary  2.019e-01  2.700e-02   7.478 7.53e-14 ***
#   contact_cellular    9.670e-01  3.325e-02  29.082  < 2e-16 ***
#   contact_telephone   8.571e-01  5.688e-02  15.070  < 2e-16 ***
#   poutcome_other      3.449e-01  5.794e-02   5.953 2.62e-09 ***
#   poutcome_success    2.396e+00  7.245e-02  33.075  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 43872  on 31647  degrees of freedom
# Residual deviance: 37356  on 31634  degrees of freedom
# AIC: 37384
# 
# Number of Fisher Scoring iterations: 5
# 
# > m2_mat_ovun
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0 8151  507
# 1 3837 1068
# 
# Accuracy : 0.6797          
# 95% CI : (0.6718, 0.6876)
# No Information Rate : 0.8839          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.1866          
# 
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.67810         
#             Specificity : 0.67993         
#          Pos Pred Value : 0.21774         
#          Neg Pred Value : 0.94144         
#              Prevalence : 0.11612         
#          Detection Rate : 0.07874         
#    Detection Prevalence : 0.36165         
#       Balanced Accuracy : 0.67901         
#                                           
#        'Positive' Class : 1               
#                                           
# > m2_mcfad_ovun
# [1] 0.1485232