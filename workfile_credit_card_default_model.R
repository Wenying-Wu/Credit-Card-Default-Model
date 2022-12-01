#  Credit Card Default Model                                                 #####          

##-----------------------------------------------------------------------------#
# .------------- Set up ---------------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# set working directory ----
setwd("D:/Machine Learning/Credit Card Default Model")
Sys.setlocale("LC_TIME", "English")

# libraries ----
library(plyr)
library(ggplot2)
library(dplyr)
library(data.table)
library(readr)
library(naniar)
library(caret)
library(parallel)
library(doParallel)
library(corrplot)
library(pROC)
library(glmnet)
library(gbm)
library(magrittr)
library(pdp)
library(lattice)
library(prediction)
library(ROCR)
library(randomForest)
library(xgboost)
library(magrittr)
library(Matrix)
library(psych)
library(tidyr)
library(stringi)
library(R6)
library(ranger)
library(lightgbm)
library(methods)
library(PerformanceAnalytics)

##-----------------------------------------------------------------------------#
# .----------- Read Raw Data --------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# raw data ----
credittrain_raw <- read.csv("AT2_credit_train.csv")
credittest_raw <- read.csv("AT2_credit_test.csv")
str(credittest_raw)

# check data ----
# summary table could check missing data and data structure,
# there are 23101 rows in the train dataset, 6899 in test dataset
str(credittrain_raw)
summary(credittrain_raw)
head(credittrain_raw)
summary_train <- apply(credittrain_raw, 2, table)
summary_test <- apply(credittest_raw, 2, table)

##-----------------------------------------------------------------------------#
# .----------- Visualization --------------------------------------------- #####          
##-----------------------------------------------------------------------------#
# make plot dataset -----
### "trainplot", change variable levels, it is more clear when doing plot
trainplot <- credittrain_raw %>%
  mutate_at(vars(3:5,7:12), as.factor) %>%  # change as factor
  mutate(AGE1 = ifelse(AGE < 25, "-25", 
                       ifelse(AGE < 35, "25-34", 
                              ifelse(AGE < 45, "35-44",
                                     ifelse(AGE < 55, "45-54",
                                            ifelse(AGE < 65, "55-64",
                                                   "65-"))))))%>%
  mutate(default = as.factor(recode(default,'Y'= 1, 'N'= 0)))


# find cut levels for numeric variable
age.cut <- apply(trainplot["AGE"], 2,
                 function (t){c(Q1 = quantile(t, .25) - 1.5 * IQR(t),
                                Q3 = quantile(t, .75) + 1.5 * IQR(t))}) %>% t()
LIMIT.cut <- apply(trainplot["LIMIT_BAL"], 2,
                   function (t){c(Q1 = quantile(t, .25) - 1.5 * IQR(t),
                                  Q3 = quantile(t, .75) + 1.5 * IQR(t))}) %>% t()
BILL.cut <- apply(trainplot %>% select(starts_with("BILL_AMT")), 2,
                  function (t){c(Q1 = quantile(t, .25) - 1.5 * IQR(t),
                                 Q3 = quantile(t, .75) + 1.5 * IQR(t))}) %>% t()


### rename levels for plot
levels(trainplot$SEX) <- c("Male", "Female", "Unknown", "Unknown", "Unknown", "Unknown")
### actually 6 levels, not 2
levels(trainplot$EDUCATION) <- c("Unknown", "Graduate School", "University", 
                                 "High school", "Others", "Unknown", "Unknown") 
### actually 7 levels, not 6
levels(trainplot$MARRIAGE) <- c("Unknown" , "Married" , "Single" ,"Others") 
### actually levels, not 3
levels(trainplot$default) <- c("non default", "default")


# Variable: ID ----
### check ID is unique
anyDuplicated(trainplot$ID)
length(unique(trainplot$ID)) 

# Variable: SEX, EDUCATION, MARRIAGE, AGE ----
SEM.col.name <- c("SEX", "EDUCATION", "MARRIAGE")
### count ----
lapply(SEM.col.name, function (sem.col.name){ 
  ggplot(data = trainplot[, SEM.col.name], aes(x = trainplot[, sem.col.name])) +
    geom_bar(stat = "count") + 
    labs(title = paste0("Bar Plot of", "\n", sem.col.name), 
         x = " ", 
         y = "Count") +
    theme_minimal()
})
### Note: there are 4 invalid entry in SEX, which should be delete when modelling 
### and there are unknown value which are not mentioned in description

### education count by sex----
ggplot(trainplot, aes(x = EDUCATION, fill = SEX)) +
  geom_bar() +
  facet_grid(~SEX) + 
  theme(legend.position = "NA") +
  scale_x_discrete(guide = guide_axis(angle = 45)) +
  labs( title = "Bar Plot of Sex", 
        subtitle = "by education", 
        x = " ", 
        y = "Count",
        fill = "SEX")

### education count by MARRIAGE----
ggplot(trainplot, aes(x = EDUCATION, fill = MARRIAGE)) +
  geom_bar() +
  facet_grid(~MARRIAGE) + 
  theme(legend.position = "NA") +
  scale_x_discrete(guide = guide_axis(angle = 45)) +
  labs( title = "Bar Plot of Marriage", 
        subtitle = "by education", 
        x = " ", 
        y = "Count",
        fill = "MARRIAGE")

# Variable: AGE ----
### count ----
ggplot(trainplot, aes(x = AGE)) +
  geom_histogram(binwidth=2,fill="black",colour="white") + 
  geom_vline(aes(xintercept = age.cut[, "Q1.25%"]), 
             colour = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = age.cut[, "Q3.75%"]), 
             colour = "red", linetype = "dashed") +
  labs(title = "Bar Plot of Age", 
       x = "Age", 
       y = "Count") +
  theme_minimal() 

### distribution ----
ggplot(trainplot, aes(x = 1:length(AGE), y = AGE)) +
  geom_point(size = 0.5) +
  geom_hline(aes(yintercept = age.cut[, "Q1.25%"]), 
             colour = "red", linetype = "dashed") +
  geom_hline(aes(yintercept = age.cut[, "Q3.75%"]), 
             colour = "red", linetype = "dashed") +
  labs(title = paste0("Distribution of Age"), 
       x = "Index", 
       y = "Age") +
  theme_minimal() 
### Note: check whether do we need to cut age 

# Variable: Limit_BAL ----
### count ----
ggplot(trainplot, aes(x = LIMIT_BAL)) +
  geom_histogram(binwidth=2000,fill="white",colour="black") + 
  geom_vline(aes(xintercept = LIMIT.cut[, "Q1.25%"]), 
             colour = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = LIMIT.cut[, "Q3.75%"]), 
             colour = "red", linetype = "dashed") +
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(title = paste0("Bar Plot of Limit Balance"), 
       x = "Limit balance", 
       y = "Count") +
  theme_minimal() 

### distribution ----
#### a. total ----
ggplot(data = trainplot, aes(x = 1:length(LIMIT_BAL), y = LIMIT_BAL)) +
  geom_point(size = 0.5) +
  geom_hline(aes(yintercept = LIMIT.cut[, "Q1.25%"]), 
             colour = "red", linetype = "dashed") +
  geom_hline(aes(yintercept=LIMIT.cut[, "Q3.75%"]), 
             colour = "red", linetype="dashed") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))+
  labs(title = "Distribution of Limit Balance", 
       x = "Index", 
       y = "Amount") +
  theme_minimal() 

### by Limit_BAL  ----
ggplot(trainplot, aes(x = default, y = LIMIT_BAL)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  scale_color_hue(labels = c("non-default", "default")) +
  labs(colour = "default",
       title  = "Limit Balance for Default and Non_default",
       subtitle = "total amount of given credit in dollars",
       y = "Amount",
       x = " ")
#### b. by SEX ----
# distribution
ggplot(trainplot, aes(x = 1:length(LIMIT_BAL), y = LIMIT_BAL, 
                      fill = SEX, color = SEX)) +
  geom_point(size = 0.5) +
  theme(legend.position = "NA") +
  facet_grid(~SEX) + 
  labs( title = "Distribution of Limit Balance", 
        subtitle = "by sex", 
        x = "Index", 
        y = "Amount")

#### & by default
ggplot(trainplot, aes(x = default, y = LIMIT_BAL)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  facet_wrap( ~ SEX) + 
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(colour = "default",
       title  = "Limit Balance for Default and Non_default",
       subtitle = "by sex",
       y = "Amount",
       x = " ")


#### c. by EDUCATION ----
ggplot(trainplot, aes(x = 1:length(LIMIT_BAL), y = LIMIT_BAL, 
                      fill = EDUCATION, color = EDUCATION)) +
  geom_point(size = 0.5) +
  theme(legend.position = "NA") +
  facet_grid(~EDUCATION) + 
  labs( title = "Distribution of Limit Balance", 
        subtitle = "by education", 
        x = "Index", 
        y = "Amount")

#### & by default
ggplot(trainplot, aes(x = default, y = LIMIT_BAL)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  facet_wrap( ~ EDUCATION, ncol = 5) + 
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(colour = "default",
       title  = "Limit Balance for Default and Non_default",
       subtitle = "by education",
       y = "Amount",
       x = " ")

#### d. by MARRIAGE ----
ggplot(trainplot, aes(x = 1:length(LIMIT_BAL), y = LIMIT_BAL, 
                      fill = MARRIAGE, color = MARRIAGE)) +
  geom_point(size = 0.5) +
  theme(legend.position = "NA") +
  facet_grid(~MARRIAGE) + 
  labs( title = "Bar Plot of Limit balance", 
        subtitle = "by marriage", 
        x = "Index", 
        y = "Amount")


#### & by default
ggplot(trainplot, aes(x = default, y = LIMIT_BAL)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  facet_wrap( ~ MARRIAGE, ncol = 4) + 
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  scale_color_hue(labels = c("non-default", "default")) +
  labs(colour = "default",
       title  = "Limit Balance for Default and Non_default",
       subtitle = "by marriage",
       y = "Amount",
       x = " ")

#### e. by AGE ----
ggplot(trainplot, aes(x = ID, y = LIMIT_BAL, 
                      fill = AGE1, color = AGE1)) +
  geom_point(size = 0.5) +
  theme(legend.position = "NA") +
  facet_grid(~AGE1) + 
  labs( title = "Bar Plot of Limit balance", 
        subtitle = "by age", 
        x = "Index", 
        y = "Amount")


#### & by default
ggplot(trainplot, aes(x = default, y = LIMIT_BAL)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  facet_wrap( ~ AGE1, ncol = 6) + 
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  scale_color_hue(labels = c("non-default", "default")) +
  labs(colour = "default",
       title  = "Limit Balance for Default and Non_default",
       subtitle = "by age",
       y = "Amount",
       x = " ")

# Variable: PAY_X ----
PAY.col.name  <- paste0("PAY_", c(0, 2:6))
### count ----
lapply(PAY.col.name, function (pay.col.name){ 
  ggplot(data = trainplot[, PAY.col.name], aes(x = trainplot[, pay.col.name])) +
    geom_bar(stat = "count") + 
    labs(title = paste0("Distribution of", "\n", pay.col.name), 
         x = " ", 
         y = "Count")+
    theme_minimal() 
})
### Note: there are a large amount of records = -2 and 0 for PAY_X, 
### which are not mentioned in description, but also found in testset
### I assume it as variables which just missing description


### Total PAY_X Summary ----
PAY <- as.data.frame(table(unlist(data.frame(trainplot[, PAY.col.name]))))

ggplot(PAY, aes(x = Var1, y= Freq)) +
  geom_bar(stat="identity") +
  geom_text(aes(label=Freq), position=position_dodge(width=0.9), vjust=-0.25)+
  labs(title = "Bar chart of Repayment status", 
       x = " Repayment Status ", 
       y = "Count")+
  theme_minimal() 

# BILL_AMTX and PAY_ATM
amount <- trainplot %>%
  select(BILL_AMT1:default) %>%
  group_by(default) %>%
  summarise_each(list(median = median, mean = mean)) %>%
  gather(-default,
         key = "name", value = "value") %>%
  separate(name, c("type", "name", "stats")) %>%
  mutate(var = paste0(type, "_", name)) %>%
  mutate(class = paste0(type, "_", stats))  %>%
  mutate(type = paste0(type, "_AMT"))%>%
  mutate(X = stri_sub(var, -1,-1))

amount_order <- c('6', '5', '4', '3', '2', '1') 

### by default ----
ggplot(subset(amount, type =="PAY_AMT"), 
       aes(x = factor(X, level = amount_order), y = value, group = default, colour = default)) +
  facet_grid(~stats) + 
  geom_line() + 
  geom_point() +  
  scale_color_hue(labels = c("non-default", "default")) +
  labs(colour = "",
       title = "Analysis Repayment Amount by Default",
       subtitle = "by mean and median",
       x = " number of past month",
       y = "Amount") +
  theme(legend.position = "bottom") 



# Variable: BILL_AMTX ----
BILL.col.name  <- paste0("BILL_AMT", c(1:6))

### count ----
lapply(BILL.col.name, function (bill.col.name){ 
  ggplot(data = trainplot[, BILL.col.name], 
         aes(x = trainplot[, bill.col.name])) +
    geom_histogram(binwidth=1000,fill="white",colour="black") + 
    geom_vline(aes(xintercept = BILL.cut[bill.col.name, "Q1.25%"]), 
               colour = "red", linetype = "dashed") +
    geom_vline(aes(xintercept = BILL.cut[bill.col.name, "Q3.75%"]), 
               colour = "red", linetype = "dashed") +
    scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) +
    labs(title = paste0("Bar Plot of", "\n", bill.col.name), 
         x = paste0(bill.col.name), 
         y = "Count")+
    theme_minimal() 
})

### distribution ----
lapply(BILL.col.name, function (bill.col.name){ 
  ggplot(data = trainplot[, BILL.col.name], 
         aes(x = 1:length(trainplot[, bill.col.name]), 
             y = trainplot[, bill.col.name])) +
    geom_point(size = 0.5) + 
    geom_hline(aes(yintercept = BILL.cut[bill.col.name, "Q1.25%"]), 
               colour = "red", linetype = "dashed") +
    geom_hline(aes(yintercept = BILL.cut[bill.col.name, "Q3.75%"]), 
               colour = "red", linetype = "dashed") +
    scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
    labs(title = paste0("Distribution of", "\n", bill.col.name), 
         x = "Index", 
         y = " ")+
    theme_minimal() 
})

# Variable: PAY_AMTX ----

### by default ----
ggplot(subset(amount, type =="BILL_AMT"), 
       aes(x = factor(X, level = amount_order), y = value, group = default, colour = default)) +
  facet_grid(~stats) + 
  geom_line() + 
  geom_point() +  
  scale_color_hue(labels = c("non-default", "default")) +
  labs(colour = "",
       title = "Analysis of Bill Amount by Default",
       subtitle = "by mean and median",
       x = " number of past month",
       y = "Amount") +
  theme(legend.position = "bottom") 


# Variable: default ----
prop.table(table(trainplot$default))

## proportions ----
ggplot(trainplot, 
       aes(x=as.factor(default), y= ..count.. / sum(..count..))) +
  geom_bar() +
  scale_y_continuous(labels = scales::percent) +
  geom_text(aes(label = scales::percent(round((..count..)/sum(..count..),2)),
                y= ((..count..)/sum(..count..))), stat="count",
            vjust = -.25) +
  labs(title = "Probability Of Defaulting Payment Next Month", 
       x =" ",
       y = "Percent")+
  theme_minimal() 


### a. by sex ----
ggplot(trainplot, aes(x=default, group = SEX)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  theme(legend.position = "NA") +
  facet_grid(~SEX)+
  scale_y_continuous(labels = scales::percent)+
  labs(title = "Probability Of Defaulting Payment Next Month", 
       subtitle = "by sex",
       x =" ",
       y = "Percent")

### b. by education ----
ggplot(trainplot, aes(x=default, group = EDUCATION)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  theme(legend.position = "NA") +
  facet_grid(~EDUCATION)+
  scale_y_continuous(labels = scales::percent)+
  labs(title = "Probability Of Defaulting Payment Next Month", 
       subtitle = "by education",
       x =" ",
       y = "Percent")

### c. by Marriage ----
ggplot(trainplot, aes(x=default, group = MARRIAGE)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  theme(legend.position = "NA") +
  facet_grid(~MARRIAGE)+
  scale_y_continuous(labels = scales::percent)+
  labs(title = "Probability Of Defaulting Payment Next Month", 
       subtitle = "by marriage",
       x =" ",
       y = "Percent")


### c. by age ----
ggplot(trainplot, aes(x=default, group = AGE1)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  theme(legend.position = "NA") +
  facet_grid(~AGE1)+
  scale_y_continuous(labels = scales::percent)+
  labs(title = "Probability Of Defaulting Payment Next Month", 
       subtitle = "by age",
       x =" ",
       y = "Percent")
## by age ----
ggplot(trainplot, aes(x = default, y = AGE1)) +
  geom_violin(aes(fill = default), trim = FALSE, alpha = 0.3) +
  geom_boxplot(aes(fill = default), width = 0.2, outlier.colour = NA) +
  theme(legend.position = "NA") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) +
  labs(colour = "default",
       title  = "Age for Default and Non_default",
       subtitle = "",
       y = "Age",
       x = " ")




##-----------------------------------------------------------------------------#
# .----------- Data Cleaning --------------------------------------------- #####          
##-----------------------------------------------------------------------------#
### 23081 rows after clean, remain 22824 row after cut, 0.9880092 remain

## data cleaning ----
## combine train and test, make easier for cleaning
data_all <- rbind(credittrain_raw[,-25], credittest_raw)
summary_data <- apply(data_all, 2, table)
data_all <- mutate_at(data_all, vars(3:5,7:12), as.factor) # change as factor
str(data_all)

xgbcol = c('SEX', 'EDUCATION', 'MARRIAGE', 
           'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6')
dummies <- dummyVars(~ SEX + EDUCATION + MARRIAGE + 
                       PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6, 
                     data = data_all)
df_all_dum <- as.data.frame(predict(dummies, newdata = data_all))
df_all_dum <- df_all_dum [,-c(3:6)]  
summary(df_all_dum)

df_all_combined <- 
  cbind(data_all[,-c(which(colnames(data_all) %in% xgbcol))],df_all_dum)

### split raw train and test ----
df = df_all_combined[df_all_combined$ID %in% credittrain_raw$ID,]
df$default <- ifelse(df$ID == credittrain_raw$ID, credittrain_raw$default, NA)

## trainset and trailset and testset???? ----
# 75% of the sample size
set.seed(1234)

df_Y <- subset(df, df$default =="Y")
df_N <- subset(df, df$default =="N")
df_sizeY <- floor(0.75 * nrow(df_Y))
df_sizeN <- floor(0.75 * nrow(df_N))
df_indicesY <- sample(seq_len(nrow(df_Y)), size = df_sizeY)
df_indicesN <- sample(seq_len(nrow(df_N)), size = df_sizeN)
train_Y <- df_Y[df_indicesY, ]
train_N <- df_N[df_indicesN, ]
train <- rbind(train_Y, train_N)
test_Y <- df_Y[-df_indicesY, ]
test_N <- df_N[-df_indicesN, ]
train <- rbind(train_Y, train_N)   # 17325 rows
test <- rbind(test_Y, test_N)   # 5776 rows
set.seed(1234)
test_size <- floor(0.75 * nrow(test))
test_indices <- sample(seq_len(nrow(test)), size = test_size)
test1 <- test[test_indices, ]
test2 <- test[-test_indices, ]



# clean and cut 
trainclean <- subset(train, train$LIMIT_BAL > 0 )
trainclean <- subset(trainclean, trainclean$AGE <= 122) # 17310 rows
### the above cleaning is base on summary table
### LIMIT_BAL: there are records are -99 in raw data, don't make sense, delete
### AGE: there are records > 122, delete ,122 is oldest people in the world
### - (base on https://en.wikipedia.org/wiki/Oldest_people)

testclean <- subset(test, test$LIMIT_BAL > 0 )
testclean <- subset(testclean, testclean$AGE <= 122) # 5771 rows

## cut outliear ----
## data bill 
BILL <- trainclean %>% select(starts_with("BILL_AMT"))  
remove.bill <- 
  lapply(colnames(BILL),
         function (BILL.col.name){
           which(BILL[, BILL.col.name] > 
                   BILL.cut[BILL.col.name, "Q3.75%"] + 220000 |
                   BILL[, BILL.col.name] < 
                   BILL.cut[BILL.col.name, "Q1.25%"])
         }
  ) %>% unlist()
BILL <- BILL[-remove.bill, ]  # remain 17115 rows, delect 210 row
totalremove <- remove.bill

## data limit
LIMIT <- trainclean[c("LIMIT_BAL")] 
remove.limit <- 
  lapply(colnames(LIMIT),
         function (LIMIT.col.name){
           which(LIMIT[, LIMIT.col.name] > 
                   LIMIT.cut[LIMIT.col.name, "Q3.75%"] + 200000 |
                   LIMIT[, LIMIT.col.name] < 
                   LIMIT.cut[LIMIT.col.name, "Q1.25%"])
         }
  ) %>% unlist()
LIMIT <- LIMIT[-remove.limit, ]   # 17305 rows, 5 outliear
totalremove <- c(totalremove, remove.limit)
traincut <- trainclean[-totalremove, ] 
## 17112, delete 2% of data after clean and cut

str(train)
## set train
y_train <- recode(train$default,'Y'=1, 'N'=0)
X_train <- select(train, -c(ID,default))

y_trainclean <- recode(trainclean$default,'Y'=1, 'N'=0)
X_trainclean <- select(trainclean, -c(ID, default))

y_traincut <- recode(traincut$default,'Y'=1, 'N'=0)
X_traincut <- select(traincut, -c(ID, default))

## set test
y_test <- recode(test$default,'Y'=1, 'N'=0)
X_test <- select(test, -c(ID, default))

y_testclean <- recode(testclean$default,'Y'=1, 'N'=0)
X_testclean <- select(testclean, -c(ID, default))

y_test1 <- recode(test1$default,'Y'=1, 'N'=0)
X_test1 <- select(test1, -c(ID, default))
y_test2 <- recode(test2$default,'Y'=1, 'N'=0)
X_test2 <- select(test2, -c(ID, default))

## set trail
X_trail = df_all_combined[df_all_combined$ID %in% credittest_raw$ID,]
y_trail <- ifelse(credittest_raw$AGE >0, 0, 1) # made one list as y
X_trail <- select(X_trail, -c(ID))
## all has 79 columns

## Count NAs ----
sapply(trainplot, function(x) sum(x == "0"))
sapply(trainplot, function(x) sum(is.na(x)))
##-----------------------------------------------------------------------------#
# .----------- Modelling ------------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# correlation ----
PAYX <- trainplot %>% select(7:12)
PAYX <- data.frame(lapply(PAYX,as.numeric))
PAYX %>%   cor() %>%
  corrplot()
chart.Correlation(PAYX)

BILX <- trainplot %>% select(13:18)
BILX <- data.frame(lapply(BILX,as.numeric))
BILX %>%   cor() %>%
  corrplot()
chart.Correlation(BILX)


# XGB setting----
new_X_train <- model.matrix(~.+0,data = X_train) 
new_X_trainclean <- model.matrix(~.+0,data = X_trainclean) 
new_X_traincut <- model.matrix(~.+0,data = X_traincut) 

new_X_test <- model.matrix(~.+0,data = X_test) 
new_X_testclean <- model.matrix(~.+0,data = X_testclean) 
new_X_trail <- model.matrix(~.+0,data = X_trail) 

dtrain <- xgb.DMatrix(data = new_X_train,label = y_train) 
dtrainclean <- xgb.DMatrix(data = new_X_trainclean,label = y_trainclean) 
dtraincut <- xgb.DMatrix(data = new_X_traincut,label = y_traincut) 

dtest <- xgb.DMatrix(data = new_X_test,label = y_test)
dtestclean <- xgb.DMatrix(data = new_X_testclean,label = y_testclean)
dtrail <- xgb.DMatrix(data = new_X_trail,label = y_trail)
new_X_test1 <- model.matrix(~.+0,data = X_test1) 
new_X_test2 <- model.matrix(~.+0,data = X_test2) 
dtest1 <- xgb.DMatrix(data = new_X_test1,label = y_test1)
dtest2 <- xgb.DMatrix(data = new_X_test2,label = y_test2)


# ...............................................................----
# 1.XGB Model ----
## Recall: 0.5046 -----
## auc : 0.8141 ----

set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.07, gamma=14, max_depth=8, min_child_weight = 10, 
               max_delta_step = 2,
               subsample=0.5, colsample_bytree=0.4)
xgb1 <- xgb.train(params = params, data = dtraincut,
                  watchlist = list(traincut=dtraincut, test=dtest),
                  nrounds = 588, print_every_n = 30, 
                  maximize = F)
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
max(a1$test_auc)
plot(a1$iter, a1$traincut_auc, col = 'blue')
lines(a1$iter, a1$test_auc, col = 'red')

a1[a1$test_auc == 0.814136,]

xgbprob1 <- predict(xgb1,dtest)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_test), positive = 1)
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtraincut), model = xgb1)

## AUC ----
auctest_xgb1 <- roc(response = y_test, 
                    predictor = xgbprob1)
auctest_xgb1
plot(auctest_xgb1)

## Write output ----
trail_xgbprob <- predict(xgb1,dtrail)
trail_xgbpred <- ifelse(trail_xgbprob > 0.5,1,0)

write.csv((test_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)),
          file="xgb_cutraw8",row.names = FALSE)
depth8 = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)



# .------------------------------------------------------------------------.----
## dateset setting 
traincut1 <- select(credittrain_raw[credittrain_raw$ID %in% traincut$ID,],-ID)
testraw <- select(credittrain_raw[credittrain_raw$ID %in% test$ID,],-ID)
testrawclean <- select(credittrain_raw[credittrain_raw$ID %in% testclean$ID,],-ID)
testrawclean$default <- ifelse(testrawclean$default == "Y", 1, 0)

# 2. GBM  ----
## use dataset without outliear
## Recall: 0.4260471 ----
## test_auc: 0.8004 ----
## train_auc: 0.8087 ----

## setting ----
cluster = makeCluster(detectCores() -1)
registerDoParallel(cluster)

##
ctrl = trainControl(method = "cv", number = 5, classProbs = TRUE, 
                    summaryFunction = twoClassSummary, allowParallel = TRUE)

gbm_grid = expand.grid(interaction.depth = c(1,2,5),
                       n.trees = c(50, 100),
                       shrinkage = c(0.01, 0.05),
                       n.minobsinnode = c(5,10))

set.seed(1234)
gbm = train(default ~ .,
            data = traincut1,
            method = "gbm",
            trControl = ctrl,
            tuneGrid = gbm_grid,
            metric = "ROC")

# plotty plot
plot(gbm)

# test AUC
testrawclean_gbm.prob = predict(gbm, testrawclean, type = "prob")
testrawclean_gbm.pred = predict(gbm, testrawclean, type = "raw")
auctestrawclean_gbm <- roc(response = testrawclean$default, 
                   predictor = testrawclean_gbm.prob$Y)
auctestrawclean_gbm
plot(auctestrawclean_gbm)
# train AUC
train_gbm.prob = predict(gbm, traincut1, type = "prob")
train_gbm.pred = predict(gbm, traincut1, type = "raw")
auctrain_gbm <- roc(response = traincut1$default, 
                    predictor = train_gbm.prob$Y)
auctrain_gbm

# confusion matrix
cm_gbm <- confusionMatrix(testrawclean_gbm.pred, 
                          as.factor(testrawclean$default), positive= "Y")
cm_gbm
cm_gbm$byClass


## feature importance ----
gbm_importance <- gbm %>%
  varImp() %>%
  extract2("importance") %>%
  tibble::rownames_to_column("variable") %>%
  arrange(desc(Overall))
gbm_importance %>%
  head(20) %>%
  ggplot(aes(Overall, reorder(variable, Overall))) +
  geom_point() +
  geom_segment(aes(y=variable, yend=variable, x=0, xend=Overall)) +
  labs(title="Variable Importance",
       y = "Variable",
       x = "importance")

densityplot(gbm, pch = "|")
par(mfrow = c(2,2))




# ...............................................................----
# 3. Random forest classfication ----
## Recall: 0.4487773 ----
## test_rf1_auc: 0.7966282 ----
## train_rf1_auc: 0.9995924 ----
traincut1rf <- traincut1
traincut1rf$default <- as.factor(traincut1rf$default)

set.seed(1234)
rf1 = randomForest(default ~., data = traincut1rf, 
                   importance=TRUE, xtest=testrawclean[,c(-24)], 
                   keep.forest=TRUE, ntree=500)
rf1

# predict
test_rf1.prob = predict(rf1, testrawclean, type = "prob")
test_rf1.pred = predict(rf1, testrawclean, type = "response")

# confusion matrix
cm_rf1 <- confusionMatrix(test_rf1.pred, 
                          as.factor(testrawclean$default), positive= "Y")
cm_rf1
cm_rf1$byClass

## AUC ----
train_rf1.prob = predict(rf1, traincut1, type = "prob")
train_rf1.pred = predict(rf1, traincut1, type = "response")
train_rf1_auc <- prediction(train_rf1.prob[, 2],traincut1$default)

train_rf1_tpr_fpr = performance(train_rf1_auc, "tpr","fpr")
train_rf1_auc = performance(train_rf1_auc, "auc")
train_rf1_auc = unlist(slot(train_rf1_auc, "y.values"))
train_rf1_auc


test_rf1_auc <- prediction(test_rf1.prob[, 2],testrawclean$default)
test_rf1_tpr_fpr = performance(test_rf1_auc, "tpr","fpr")
test_rf1_auc = performance(test_rf1_auc, "auc")
test_rf1_auc = unlist(slot(test_rf1_auc, "y.values"))
test_rf1_auc

## feature importance ----
# Plot of most important variables
varImpPlot(rf1)

# ...............................................................----

# 4. lasso  ----
## use dataset without outliear
## Recall: 0.0013089005 ---- 
## F1: 0.0025974026  ----
## test_auc: 0.6551012 ----
## train_auc: 0.6505553 ----
set.seed(1234)
x <-  model.matrix(~ ., traincut1)

# Create model, Alpha = 1 specifies lasso regression
las <-  cv.glmnet(x, y_traincut, family = 'binomial', alpha = 1)
las

# plotty plot
plot(las)

summary(las)

# Predict the model
testrawcleanprediction <-  predict(las$glmnet.fit, 
                                  newx = model.matrix(~ ., testrawclean), 
                                  type = "class", s = las$lambda.min)
testrawcleanprobability <-  predict(las$glmnet.fit, 
                                   newx = model.matrix(~ ., testrawclean), 
                                   type = "response", s = las$lambda.min)

a <- testrawcleanprediction
# Calculate prediction and probability
# confusion matrix
cm_las <-  confusionMatrix(as.factor(testrawcleanprediction), 
                           as.factor(testrawclean$default), positive = "1")
cm_las
# precision, recall, F1, AUC
cm_las$byClass 

## AUC ----
trainrawcleanprediction <-  predict(las$glmnet.fit, 
                                   newx = model.matrix(~ ., traincut1), 
                                   type = "class", s = las$lambda.min)
trainrawcleanprobability <-  predict(las$glmnet.fit, 
                                    newx = model.matrix(~ ., traincut1), 
                                    type = "response", s = las$lambda.min)
train_las_auc <- prediction(trainrawcleanprobability,traincut1$default)

train_las_tpr_fpr = performance(train_las_auc, "tpr","fpr")
train_las_auc = performance(train_las_auc, "auc")
train_las_auc = unlist(slot(train_las_auc, "y.values"))
train_las_auc


test_las_auc <- prediction(testrawcleanprobability,testrawclean$default)
test_las_tpr_fpr = performance(test_las_auc, "tpr","fpr")
test_las_auc = performance(test_las_auc, "auc")
test_las_auc = unlist(slot(test_las_auc, "y.values"))
test_las_auc

# ...............................................................----


# 5. glm -cut variable - glm1 ----
## use dataset without outliear
## Recall: 0.25327225  ---- 
## F1: 0.36927481 ----
## test_auc: 0.7209168 ----
## train_auc: 0.7179081 ----
trainlm <- traincut1
trainlm$default <- ifelse(trainlm$default == "Y", 1, 0)

set.seed(1234)
glm <- glm(default ~ ., trainlm, family = "binomial")

glm
summary(glm)

set.seed(1234)
glm1 <- glm(default ~ LIMIT_BAL + SEX + MARRIAGE + AGE + 
              PAY_0 + PAY_3 + PAY_5 + PAY_AMT1 + PAY_AMT3 + 
              PAY_AMT4 + PAY_AMT5, trainlm, family = "binomial")
summary(glm1)
# predict
test_glm1.prob = predict(glm1, testrawclean, type = "response")
test_glm1.pred = ifelse(test_glm1.prob >= 0.5, "Y", "N")
str(testlm)
# confusion matrix
cm_glm1 <- confusionMatrix(as.factor(test_glm1.pred), 
                           as.factor(testrawclean$default), positive = "Y")
cm_glm1
cm_glm1$byClass


## AUC ----
train_glm1.prob = predict(glm1, traincut1, type = "response")
train_glm1.pred = ifelse(train_glm1.prob >= 0.5, "Y", "N")
train_glm1_auc <- prediction(train_glm1.prob,traincut1$default)

train_glm1_tpr_fpr = performance(train_glm1_auc, "tpr","fpr")
train_glm1_auc = performance(train_glm1_auc, "auc")
train_glm1_auc = unlist(slot(train_glm1_auc, "y.values"))
train_glm1_auc

# ...............................................................----
###....... Further test appendix .............----
# ...............................................................----
# XGE Model ----
# 1. raw train raw test  -----
## all variable
## auc : 0.815915 ----
## setting

set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc", min_child_weight=5,
               eta=0.05, gamma=6, max_depth=9,  max_delta_step=0,
               subsample=0.9, colsample_bytree=0.9)

xgb1 <- xgb.train(params = params, data = dtrain,
                  watchlist = list(train=dtrain, test=dtest),
                  nrounds = 213, print_every_n = 30, 
                  early_stop_round = 10, 
                  maximize = F)
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
max(a1$test_auc)
plot(a1$iter, a1$train_auc, col = 'blue')
lines(a1$iter, a1$test_auc, col = 'red')

a1[a1$test_auc == 0.815915,]


xgbprob1 <- predict(xgb1,dtest)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_test), positive = 1)
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtrain), model = xgb1)

## AUC ----
auctest_xgb1 <- roc(response = y_test, 
                    predictor = xgbprob1)
auctest_xgb1
plot(auctest_xgb1)

## Write output ----
trail_xgbprob <- predict(xgb1,dtrail)
trail_xgbpred <- ifelse(trail_xgbprob > 0.5,1,0)

write.csv((test_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)),
          file="test_xgb1-2",row.names = FALSE)

# ...............................................................----
# XGE Model ----
# 2. clean train raw test  -----
## all variable
## auc : 0.81767 ----

## setting

set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.1, gamma=10, max_depth=7, min_child_weight = 1, 
               subsample=0.8, colsample_bytree=0.9)
xgb1 <- xgb.train(params = params, data = dtrainclean,
                  watchlist = list(trainclean=dtrainclean, test=dtest),
                  nrounds = 1000, print_every_n = 30, 
                  maximize = F)
xgb1
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
max(a1$test_auc)
plot(a1$iter, a1$trainclean_auc, col = 'blue')
lines(a1$iter, a1$test_auc, col = 'red')

a1[a1$test_auc == 0.81767,]

xgbprob1 <- predict(xgb1,dtest)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_test), positive = 1)
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtrainclean), model = xgb1)

## AUC ----
auctest_xgb1 <- roc(response = y_test, 
                    predictor = xgbprob1)
auctest_xgb1
plot(auctest_xgb1)

## Write output ----
trail_xgbprob <- predict(xgb1,dtrail)
trail_xgbpred <- ifelse(trail_xgbprob > 0.5,1,0)

write.csv((test_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)),
          file="testraw_xgb1",row.names = FALSE)


# ...............................................................----
# XGE Model ----
# 3. clean train clean test  -----
## all variable
## auc : 0.817539 ----
## setting

set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.1, gamma=10, max_depth=7, min_child_weight = 1, 
               subsample=0.8, colsample_bytree=0.9)
xgb1 <- xgb.train(params = params, data = dtrainclean,
                  watchlist = list(trainclean=dtrainclean, testclean=dtestclean),
                  nrounds = 1000, print_every_n = 30, 
                  maximize = F)
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
max(a1$testclean_auc)
plot(a1$iter, a1$trainclean_auc, col = 'blue')
lines(a1$iter, a1$testclean_auc, col = 'red')

a1[a1$testclean_auc == 0.817539,]

xgbprob1 <- predict(xgb1,dtestclean)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_testclean), positive = "1")
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtrainclean), model = xgb1)
xgb.plot.importance(imp1)

## AUC ----
auctestclean_xgb1 <- roc(response = y_testclean, 
                         predictor = xgbprob1)
auctestclean_xgb1
plot(auctestclean_xgb1)

## Write output ----
trail_xgbprob <- predict(xgb1,dtrail)
trail_xgbpred <- ifelse(trail_xgbprob > 0.5,1,0)

write.csv((testclean_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)),
          file="testclean_xgb1",row.names = FALSE)
gb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)

# ...............................................................----

# XGE Model ----
# 4. cut train / clean test ....-----
## all variable
## auc : 0.8124 ----
set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.15, gamma=15, max_depth=6,  min_child_weight = 1,
               subsample=0.8, colsample_bytree=0.9)
xgb1 <- xgb.train(params = params, data = dtraincut,
                  watchlist = list(traincut=dtraincut, testclean=dtestclean),
                  nrounds = 694, print_every_n = 30, 
                  maximize = F)
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
plot(a1$iter, a1$traincut_auc, col = 'blue')
lines(a1$iter, a1$testclean_auc, col = 'red')
a1[a1$testclean_auc == max(a1$testclean_auc),]

xgbprob1 <- predict(xgb1,dtestclean)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_testclean), positive = 1)
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtraincut), model = xgb1)

## AUC ----
auctestclean_xgb1 <- roc(response = y_testclean, 
                         predictor = xgbprob1)
auctestclean_xgb1
plot(auctestclean_xgb1)

## Write output ----
trail_xgbprob <- predict(xgb1,dtrail)
trail_xgbpred <- ifelse(trail_xgbprob > 0.5,1,0)

write.csv((testclean_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob)),
          file="cuttestcl",row.names = FALSE)

# ...............................................................----
# XGE Model ----
# 5. clean train cut var clean test ....... -----
## not good, shouldn't cut variable
## auc : 0.999931 ----
## setting
Varim <- imp1[with(imp1,order(-Frequency)),] 
Varim <- Varim[1:40,]
Varim <- Varim$Feature
X_traincleanim <- data.frame(X_trainclean[, Varim])
new_X_traincleanim <- model.matrix(~.+0,data = X_traincleanim) 
dtraincleanim <- xgb.DMatrix(data = new_X_traincleanim,label = y_trainclean) 

X_testcleanim <- data.frame(X_testclean[, Varim])
new_X_testcleanim <- model.matrix(~.+0,data = X_testcleanim) 
dtestcleanim <- xgb.DMatrix(data = new_X_testcleanim,label = y_testclean) 

X_trailim <- data.frame(X_trail[, Varim])
new_X_trailim <- model.matrix(~.+0,data = X_trailim) 
dtrailim <- xgb.DMatrix(data = new_X_trailim,label = y_trail) 


set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.1, gamma=0, max_depth=10, min_child_weight = 1, 
               subsample=0.9, colsample_bytree=0.9)
xgb1 <- xgb.train(params = params, data = dtraincleanim,
                  watchlist = list(traincleanim=dtraincleanim, traincleanim=dtraincleanim),
                  nrounds = 391, print_every_n = 30, 
                  maximize = F)
# training & test error plot
a1 <- data.frame(xgb1$evaluation_log)
max(a1$traincleanim_auc)
plot(a1$iter, a1$traincleanim_auc, col = 'blue')
lines(a1$iter, a1$traincleanim_auc, col = 'red')

a1[a1$traincleanim_auc == 0.860389,]

xgbprob1 <- predict(xgb1,dtraincleanim)
xgbpred1 <- ifelse(xgbprob1 > 0.5,1,0)
# confusion matrix
cm_xgb1 <- confusionMatrix(as.factor(xgbpred1), as.factor(y_trainclean), positive = 1)
cm_xgb1
cm_xgb1$byClass

# importance
imp1 <- xgb.importance(colnames(dtraincleanim), model = xgb1)
xgb.plot.importance(imp1)

## AUC ----
auctraincleanim_xgb1 <- roc(response = y_trainclean, 
                            predictor = xgbprob1)
auctraincleanim_xgb1
plot(auctraincleanim_xgb1)

## Write output ----
trailim_xgbprob <- predict(xgb1,dtrailim)
trailim_xgbpred <- ifelse(trailim_xgbprob > 0.5,1,0)

write.csv((traincleanim_xgb = data.frame(ID=credittest_raw$ID, default = trailim_xgbprob)),
          file="traincleanim_xgb1",row.names = FALSE)


# ...............................................................----
# XGE Model 2:  -----
# 6. cut variable PAYX test----
## auc : 0.814 ----
## setting
X_train2 <- X_train[, -c(9:14)] 
X_test2 <- X_test[, -c(9:14)] 
X_trail2 <- X_trail[, -c(9:14)] 
new_X_train2 <- model.matrix(~.+0,data = X_train2) 
new_X_test2 <- model.matrix(~.+0,data = X_test2) 
new_X_trail2 <- model.matrix(~.+0,data = X_trail2) 

dtrain2 <- xgb.DMatrix(data = new_X_train2,label = y_train) 
dtest2 <- xgb.DMatrix(data = new_X_test2,label = y_test)
dtrail2 <- xgb.DMatrix(data = new_X_trail2,label = y_trail)

set.seed(1234)
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eval_metric = "auc",
               eta=0.1, gamma=2, max_depth=6,  
               subsample=0.8, colsample_bytree=0.9)
xgb2 <- xgb.train(params = params, data = dtrain2,
                  watchlist = list(train=dtrain2, test=dtest2),
                  nrounds = 84, print_every_n = 30, 
                  early_stop_round = 10, 
                  maximize = F)

# training & test error plot
a2 <- data.frame(xgb2$evaluation_log)
max(a2$test_auc)

plot(a2$iter, a2$train_auc, col = 'blue')
lines(a2$iter, a2$test_auc, col = 'red')
a2[a2$test_auc == 0.814,]


xgbprob2 <- predict(xgb2,dtest2)
xgbpred2 <- ifelse(xgbprob2 > 0.5,1,0)
# confusion matrix
cm_xgb2 <- confusionMatrix(as.factor(xgbpred2), as.factor(y_test), positive = 1)
cm_xgb2
cm_xgb2$byClass

# importance
imp2 <- xgb.importance(colnames(dtrain2), model = xgb2)

## AUC ----
auctest_xgb2 <- roc(response = y_test, 
                    predictor = xgbprob2)
auctest_xgb2
plot(auctest_xgb2)

## Write output ----
trail_xgbprob2 <- predict(xgb2,dtrail2)
trail_xgbpred2 <- ifelse(trail_xgbprob2 > 0.5,1,0)

write.csv((test_xgb = data.frame(ID=credittest_raw$ID, default = trail_xgbprob2)),
          file="test_xgb2-2",row.names = FALSE)

# ...............................................................----

# 7. GBM - cut payment ----
## Recall: 0.4273560  ----
## F1: 0.5372275 ----
## test_gbm.cutp_auc: 0.8005 ----
## train_gbm.cutp_auc: 0.8085 ----
traincut1p <- data.frame(traincut1[, -c(18:23)])

set.seed(1234)
gbm.cutp = train(default ~ .,
                 data = traincut1p,
                 method = "gbm",
                 trControl = ctrl,
                 tuneGrid = gbm_grid,
                 metric = "ROC")

# plotty plot
plot(gbm.cutp)


# test AUC
testrawclean_gbm.cutp.prob = predict(gbm.cutp, testrawclean, type = "prob")
testrawclean_gbm.cutp.pred = predict(gbm.cutp, testrawclean, type = "raw")
auctestrawclean_gbm.cutp <- roc(response = testrawclean$default, 
                                predictor = testrawclean_gbm.cutp.prob$Y)
auctestrawclean_gbm.cutp
plot(auctestrawclean_gbm.cutp)
# train AUC
train_gbm.cutp.prob = predict(gbm.cutp, traincut1, type = "prob")
train_gbm.cutp.pred = predict(gbm.cutp, traincut1, type = "raw")
auctrain_gbm.cutp <- roc(response = traincut1$default, 
                         predictor = train_gbm.cutp.prob$Y)
auctrain_gbm.cutp

# confusion matrix
cm_dt <- confusionMatrix(test_gbm.cutp.cutp.pred, 
                         as.factor(testrawclean$default), positive= "Y")
cm_dt
cm_dt$byClass

## Write output ----
write.csv((test_gbm.cutp.cutp = data.frame(ID=testrawclean$ID, default = test_gbm.cutp.cutp.prob[, 2])),
          file="test_gbm.cutp",row.names = FALSE)


# ...............................................................----
