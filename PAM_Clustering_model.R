rm(list = ls())

df <- read.csv("C:\\Users\\nehas\\Documents\\GT 2023\\MGT 6203\\Project\\bank-full.csv", sep=';')

library(dplyr)
library(tidyverse)
library(scales)
library(caret)
library(fastDummies)
library(corrplot)
library(cluster)
library(ggplot2) 

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


# #making age groups (did not improve anything )
# as.data.frame(table(df$age))
# df[df$age >= 11 & df$age <= 20, "age_group"] <- "20"
# df[df$age >= 21 & df$age <= 30, "age_group"] <- "30"
# df[df$age >= 31 & df$age <= 40, "age_group"] <- "40"
# df[df$age >= 41 & df$age <= 50, "age_group"] <- "50"
# df[df$age >= 51 & df$age <= 60, "age_group"] <- "60"
# df[df$age >= 61 & df$age <= 70, "age_group"] <- "70"
# df[df$age >= 71 & df$age <= 80, "age_group"] <- "80"
# df[df$age >= 81 & df$age <= 95, "age_group"] <- "95"
# as.data.frame(table(df$age_group))
# #defined by upper limit 
# df$age_group <- as.integer(df$age_group)
y_job = data.frame(table(df$job, df$y))
y_job$percent = ifelse(y_job$Var2=="yes",round((y_job$Freq[(nrow(y_job)/2+1):nrow(y_job)]/sum(y_job$Freq[(nrow(y_job)/2+1):nrow(y_job)]))*100,1),round((y_job$Freq[1:(nrow(y_job)/2)]/sum(y_job$Freq[1:(nrow(y_job)/2)]))*100,1))
y_job$label = paste0(y_job$Freq,"(",y_job$percent,"%)")

ggplot(y_job, aes(x = Var2, y = percent, fill = Var1, label = label)) +
  geom_bar(stat = "identity") +
  geom_text(position = position_stack(vjust=0.5))
#include admin, management, blue collar, technician 

#since there are 11 different factors I will try to recude them to other and not working 
#other will include: housemaid, self-employed and services, entrepreneur 
#not-working will include: retired, student and unemployed 
as.data.frame(table(df$job))
df$job <- gsub("housemaid", "other", df$job)
df$job <- gsub("self-employed", "other", df$job)
df$job <- gsub("services", "other", df$job)
df$job <- gsub("entrepreneur", "other", df$job)
df$job <- gsub("student", "notworking", df$job)
df$job <- gsub("retired", "notworking", df$job)
df$job <- gsub("unemployed", "notworking", df$job)
as.data.frame(table(df$job))
df <- df[df$job != "notworking" & df$job != "other", ]
table(df$y)
#12.1% yes


#CHANGE FROM CHR TO FACTOR 
df$job = as.factor(df$job)
df$marital = as.factor(df$marital)
df$education = as.factor(df$education)
df$default = as.factor(df$default)
df$housing = as.factor(df$housing)
df$loan = as.factor(df$loan)
df$y = as.factor(df$y)

#---------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------

#CLASSIFICATION 
#source:https://github.com/tranthibaothuy/marketing-data-analysis/blob/master/Customer%20Segmentation%20with%20PAM_post.R
#source:https://mymarketinganalytics.wordpress.com/2017/12/22/customer-segmentation-with-the-cluster-analysis-by-pam-in-r/

#CREATION OF DUMMY VARS
dummy_data <- fastDummies::dummy_cols(df[,c(1:12)],remove_first_dummy = FALSE)
str(dummy_data)

df1 <- dummy_data[ ,  !names(dummy_data) %in%
                     c("job", "marital", "education", "default", "housing","loan","month")]

df2 <- cbind(df1,y = df$y )
str(df2)

new_colnames <- gsub("-", "_", colnames(df2))
colnames(df2) <- new_colnames

df_cluster = subset(df2, select = -c(balance, day, campaign, previous,
                                     default_no, default_yes, housing_no, housing_yes,
                                     loan_no, loan_yes, month_apr, month_aug, month_dec,
                                     month_feb, month_jan, month_jul, month_jun, month_mar,
                                     month_may, month_nov, month_oct, month_sep
))

df_cluster <- df_cluster %>% relocate(y)
#convert int to factor 
cols <- c(3:12)
df_cluster[cols] <- lapply(df_cluster[cols], factor) 

summary(df_cluster)

#take subset of data since number of observations is a lot 
set.seed(123)
index2<-createDataPartition(df_cluster$y,p=0.3,list=FALSE)
df_cluster_subset<-df_cluster[index2,]
table(df_cluster_subset$y)
summary(df_cluster_subset)

gower_dist <- daisy(df_cluster_subset[, -1], metric = "gower",
                    type = list(logratio = 3))
summary(gower_dist)

gower_mat <- as.matrix(gower_dist)

# Output most similar pair
df_cluster_subset[
  which(gower_mat == min(gower_mat[gower_mat != min(gower_mat)]),
        arr.ind = TRUE)[1, ], ]

# Output most dissimilar pair
df_cluster_subset[
  which(gower_mat == max(gower_mat[gower_mat != max(gower_mat)]),
        arr.ind = TRUE)[1, ], ]

#----------------------------------
# Selecting the number of clusters
# Calculate silhouette width for many k using PAM
# This step may take 10 minutes or more to complete 
# (depending on the number of Clusters)
start_time <- Sys.time()
sil_width <- c(NA)
for(i in 2:10){
  pam_fit <- pam(gower_dist, diss = TRUE, k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
end_time <- Sys.time()
time.elapse <- (end_time - start_time)
print(time.elapse)
# Time difference of 7.400829 mins

# Plot sihouette width (higher is better)

plot(1:10, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:10, sil_width)


# Cluster Interpretation

pam_fit <- pam(gower_dist, diss = TRUE, k = 6)

pam_results <- df_cluster_subset %>%
  dplyr::select(-y) %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))

pam_results$the_summary

# Show the Medoids
# interpretation is that the medoids serve as exemplars of each cluster
df_cluster_subset[pam_fit$medoids, ]

# Mosaic Plot
# response  Response to term deposit offer (yes, no)
# ----------------------------------------------------
# add the cluster membership information into dataset 
df_cluster_subset$cluster <- pam_fit$clustering

with(df_cluster_subset, print(table(cluster, y)))

mosaicplot(~cluster + y, data = df_cluster_subset,
           main =list("Subcribe the Term of Deposit",col="gray20"),
           xlab="Cluster", ylab="Subcribe",col = c("gray70","mediumseagreen"),
           type = "deviance", border = NA)
#----------------------------------------------------------
# Cluster Interpretation
# Via Descriptive Statistics
#----------------------------------------------------------
# compute percentage of yes responses to term deposit offer
response_table <- table(df_cluster_subset$cluster, df_cluster_subset$y)
cat("\nPercentage Responses\n")
for (i in 1:6)
  cat("\n", toupper(letters[i]),
      round(100 * response_table[i,2] /
              sum(response_table[i,]), digits = 1))

# compute the percentage of the customers in each cluster
print(round(100 * table(df_cluster_subset$cluster) / nrow(df_cluster_subset), digits = 1))
