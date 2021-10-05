#global configurations
Sys.setlocale("LC_TIME", "C")

#packages
library(data.table)
library(plyr)
library(readr)
library(stringr)
library(dplyr)
library(ggQC)
library(fastDummies)
library(cluster)
library(factoextra)
library(purrr)
library(mclust)
library(xgboost)

##data reading
#reading structure from data
dt <- fread('data-csv/purchase-card-transactions-201612.csv', nrows = 0)

#getting file names
files <- list.files('data-csv', pattern = '..csv', full.names = TRUE)

#standardizing column names: no spaces and uppercase
column.names <- toupper(gsub(" ", "_", colnames(dt)))

#read all files and store them in a list
list.dt <- lapply(files, fread, select = colnames(dt), col.names = column.names)

#collapse all elements in one data.table
dt <- as.data.table(ldply(list.dt, data.frame))

#remove unnecessary/remaining environment variables and cleaning garbage in ram
rm(list.dt, files, column.names)
gc()

#data cleaning
#extract just the numeric part from the card number as key
dt[ , CARD_NUMBER := str_pad(parse_number(dt[ , CARD_NUMBER]), width = 4, side = 'left', pad = "0")]

#transform the transaction date from string to date format
dt[ , TRANS_DATE := as.Date(dt[ , TRANS_DATE], format = "%d/%m/%y")]

#dropping transaction codes because we are keeping the description
dt[ , c("TRANS_CAC_CODE_1", "TRANS_CAC_CODE_2", "TRANS_CAC_CODE_3") := NULL]

#dropping TRANS_VAT_DESC because there is not metadata and we cannot infer its meaning
dt[ , TRANS_VAT_DESC := NULL]

#summary
summary(dt)

#just 1 NA's in TRANS_DATE, so we remove it
dt <- dt[!is.na(TRANS_DATE)]

#original_gross_amt is a character column, we cast it as numeric
as.numeric(dt[ , ORIGINAL_GROSS_AMT])

#it warns as it has introduced NAs so something must be happening, we check the nans
dt[which(is.na(as.numeric(dt[ , ORIGINAL_GROSS_AMT])))]

#it is the thousands separator, we replace it and cast again as numeric
dt[ , ORIGINAL_GROSS_AMT := as.numeric(gsub(",", "", ORIGINAL_GROSS_AMT))]

##data exploration
#we explore our only numeric variable
hist(dt[ , ORIGINAL_GROSS_AMT])

#we see that we got outliers, so we limit our graph on the quantiles 1% and 99% (extreme values)
ext_q <- quantile(dt[ , ORIGINAL_GROSS_AMT], probs = c(0.01, 0.99))
hist(dt[between(ORIGINAL_GROSS_AMT, ext_q[1], ext_q[2]), ORIGINAL_GROSS_AMT])

#exploring number of distinct values in each column
#we declare a function for unique values
f <- function(x){
  length(unique(x))
}

#we apply the function to the margin 2 (columns)
apply(dt, MARGIN = 2, f)

#we see that 
#we have 370 different days with transactions
#18083 different monetary amounts in transactions
#6268 different merchants
#1028 different card numbers (assuming 1028 different clients)
#125 grouped type of business according to desc 1
#888 grouped type of business according to desc 2
#13 grouped type of business according to directorate

#we show the 20 most frequent desc 2
head(dt[ , .N, TRANS_CAC_DESC_2][order(N, decreasing = TRUE)], 20)

#we see that it is somehow related to institutions or schools but since we got no metadata and the are 88 categories we chose to drop it
dt[ , TRANS_CAC_DESC_2 := NULL]

#we show the 20 most frequent merchant names
head(dt[ , .N, MERCHANT_NAME][order(N, decreasing = TRUE)], 20)

#we see that amazon has more than 5 variations in its name, so we group it
dt[grepl('amazon', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'amazon']

#we do the same for other similar cases
dt[grepl('asda', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'asda']
dt[grepl('travelodge', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'travelodge']
dt[grepl('argos', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'argos']

#everything else below 500 trx is grouped in a bag
head(dt[ , .N, MERCHANT_NAME][order(N, decreasing = TRUE)], 20)
common.merchants <- c('amazon', 'travelodge', 'asda', 'argos', 'post office counter')
dt[!dt[ , MERCHANT_NAME] %in% common.merchants, MERCHANT_NAME := 'other']

#we show the 20 most frequent desc 1
head(dt[ , .N, TRANS_CAC_DESC_1][order(N, decreasing = TRUE)], 20)

#we keep the groups with more than 5% of total transactions and the rest is grouped in a bag
(gt5pct <- dt[ , .N, TRANS_CAC_DESC_1][order(N, decreasing = TRUE)][N > 0.05*nrow(dt), ][ , TRANS_CAC_DESC_1])
dt[!dt[ , TRANS_CAC_DESC_1] %in% gt5pct, TRANS_CAC_DESC_1 := 'other']

#we show the 10 most frequent directorate
dt[ , DIRECTORATE := toupper(DIRECTORATE)]
head(dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)], 10)

#we keep the groups with more than 5% of total transactions and the rest is grouped in a bag
(gt5pct <- dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)][N > 0.05*nrow(dt), ][ , DIRECTORATE])
dt[!dt[ , DIRECTORATE] %in% gt5pct, DIRECTORATE := 'other']
head(dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)], 10)

##feature engineering trx-level
#we just have 6 columns and one is the key (card_number) so just 5 features, we need to create more in order to make clusters
#extract the day as a variable
dt[ , DAY := as.numeric(substr(x = TRANS_DATE, start = 9, stop = 10))]

#extract the weekday as a variable
dt[ , WEEKDAY := weekdays(dt[ , TRANS_DATE])]

#extract the months as a variable
dt[ , MONTH := as.numeric(substr(x = TRANS_DATE, start = 6, stop = 7))]

#chargebacks
dt[ , CHARGEBACK := ifelse(ORIGINAL_GROSS_AMT < 0, 1, 0)]

#amounts to positive
dt[ , POSITIVE_AMT := ifelse(CHARGEBACK == 1, -1*ORIGINAL_GROSS_AMT, ORIGINAL_GROSS_AMT)]

#outliers
iqr <- quantile(dt[ , POSITIVE_AMT], probs = c(0.25, 0.75))
dt[ , OUTLIER := ifelse(!between(POSITIVE_AMT, iqr[1]-1.5*QCrange(iqr), iqr[2]+1.5*QCrange(iqr)), 1, 0)]

#extreme values
dt[ , EXTREME_VALUE := ifelse(!between(POSITIVE_AMT, iqr[1]-3*QCrange(iqr), iqr[2]+3*QCrange(iqr)), 1, 0)]

#tail values
tails <- quantile(dt[ , POSITIVE_AMT], probs = c(0.025, 0.975))
dt[ , TAIL_VALUE := ifelse(!between(POSITIVE_AMT, tails[1], tails[2]), 1, 0)]

#over the median (otm)
median.value <- median(dt[ , POSITIVE_AMT])
dt[ , OTM := ifelse(POSITIVE_AMT > median.value, 1, 0)]

#transactions around payday
paydays <- c(1, 2, 14, 15, 16, 28, 29, 30, 31)
dt[ , PAYDAY_TRX := ifelse(DAY %in% paydays, 1, 0)]

#transactions on weekends
weekend.days <- c('Saturday', 'Sunday')
dt[ , WEEKEND_TRX := ifelse(WEEKDAY %in% weekend.days, 1, 0)]

summary(dt)

##feature engineering client-level
#we are going to create features grouping information by client
#first, we need to create dummies from the categorical variables
cols.for.dummy <- c('MERCHANT_NAME', 'TRANS_CAC_DESC_1', 'DIRECTORATE', 'WEEKDAY')
dt.dummies <- dummy_cols(dt, select_columns = cols.for.dummy) #, remove_selected_columns = TRUE)
summary(dt.dummies)


getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

dt.grouped <- dt.dummies[ , .(NUM_TRX = .N, AVG_TRX = mean(POSITIVE_AMT), 
                MAX_TRX = max(POSITIVE_AMT), NUM_CHARGEBACKS = sum(CHARGEBACK),
                PCT_CHARGEBACKS = sum(CHARGEBACK)/.N, 
                AVG_AMT_CHARGEBACKS = mean(POSITIVE_AMT*CHARGEBACK),
                PCT_AMT_CHARGEBACKS = sum(POSITIVE_AMT*CHARGEBACK)/sum(POSITIVE_AMT),
                NUM_OUTLIER = sum(OUTLIER), PCT_OUTLIER = sum(OUTLIER)/.N,
                NUM_XTRM_VALUE = sum(EXTREME_VALUE), PCT_XTRM_VALUE = sum(EXTREME_VALUE)/.N,
                NUM_TAIL_VALUE = sum(TAIL_VALUE), PCT_TAIL_VALUE = sum(TAIL_VALUE)/.N,
                NUM_OTM = sum(OTM), PCT_OTM = sum(OTM)/.N, 
                NUM_PAYDAY_TRX = sum(PAYDAY_TRX), PCT_PAYDAY_TRX = sum(PAYDAY_TRX)/.N,
                NUM_WEEKEND_TRX = sum(WEEKEND_TRX), PCT_WEEKEND_TRX = sum(WEEKEND_TRX)/.N,
                MODE_MERCHANT = as.factor(getmode(MERCHANT_NAME)), 
                MODE_CAC_1 = as.factor(getmode(TRANS_CAC_DESC_1)),
                MODE_DIRECT = as.factor(getmode(DIRECTORATE)), 
                MODE_DAY = as.factor(getmode(DAY)), 
                MODE_MONTH = as.factor(getmode(MONTH))), 
                CARD_NUMBER]

summary(dt.grouped)


key <- dt.grouped[ , CARD_NUMBER]
dt.grouped[ , CARD_NUMBER := NULL]

#we create a distance/dissimilarity matrix
set.seed(100)
gower_dist <- daisy(as.data.frame(dt.grouped),
                    metric = "gower")

#elbow method
wss <- function(k, x) {
  kmeans(x = x, k, nstart = 10)$tot.withinss
}

k <- 1:7
wss_val <- map_dbl(k, wss, gower_dist)

plot(k, wss_val, type = "b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K", ylab = "Total within-clusters sum of squares")
#inflection points at 2 (high) and 4 (2nd high)

#silhouette method function
silhouette_score <- function(k, x){
  km <- kmeans(x = x, centers = k, nstart = 10)
  ss <- silhouette(km$cluster, dist(x))
  mean(ss[ , 3])
}
k <- 2:7
avg_sil <- sapply(k, silhouette_score, gower_dist)
plot(k, avg_sil, type = 'b', pch = 19, frame = FALSE,
     xlab = 'Number of clusters K', ylab = 'Average Silhouette Scores')
#peaks at 2 (high) and 4 (2nd high)

#we proceed to use K=2
km <- kmeans(x = gower_dist, centers = 2)
km$size
prop.table(km$size)
#unbalanced clusters, probably anomalies detected

#in order to know with variables are the most important in determining the cluster belonging we train a supervised model
#first we must convert categorical variables into dummies
cols.for.dummy.2 <- c('MODE_MERCHANT', 'MODE_CAC_1', 'MODE_DIRECT', 'MODE_DAY', 'MODE_MONTH')
dt.dummies.2 <- dummy_cols(dt.grouped, select_columns = cols.for.dummy.2, remove_selected_columns = TRUE)

#we bind the "target" column which is the cluster (1 and 2) converted to 0 and 1
dt.sup <- cbind(dt.dummies.2, cluster = km$cluster-1)

#we train a classification model, in this case it is a binary classification since we have 2 clusters
#but in order to make it for general purposes (more than 2 clusters)
#we train a multinomial xgboost model

#number of classes
k <- length(km$size)
#definition of the multinomial model
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = k)
#number of iterations
nround <- 500 
#number of folds for the cross validation
cv.nfold <- 5

#we declare the train data matrix
train_matrix <- xgb.DMatrix(data = as.matrix(dt.sup[ , -80]), label = dt.sup$cluster)

#we do a cross validation to get the best iteration parameter
set.seed(5)
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE,
                   early_stopping_rounds = 50)

#we train the xgboost with the best number of iterations
set.seed(10)
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = cv_model$best_iteration)

#we plot the top 10 variables
importance <- xgb.importance(feature_names = colnames(train_matrix), model = bst_model)
xgb.plot.importance(importance_matrix = importance, top_n = 10, measure = 'Frequency')
(imp_features <- head(importance[order(Frequency, decreasing = TRUE)], 10)[ , Feature])

imp_features <- c(imp_features, 'cluster')
#we use these features to see what characterizes the clusters
dt.sup[ , ..imp_features][ , lapply(.SD, mean), cluster][order(cluster)]

#we can see a lot of difference in 
#pct_chargebacks, with the cluster 1 with more than 80% negative trx
#avg_trx, with a ticket per transaction of over 100 times the cluster 0
#pct_outlier, more than 80% of trx amt qualified as outliers
#mode_month_5, in 76% of cases the month with most trx was may (wierd)

#card numbers from people that have their trx mode on may
index <- which((dt.sup[ , MODE_MONTH_5] == 1) & unname(km$cluster-1 == 1))
anomaly_card_number <- key[index]

#transaction dates of that clients
table(dt[CARD_NUMBER %in% anomaly_card_number, TRANS_DATE])
#all their trx are placed on may, none other month and it is their only trx in the whole year
#it could be a fraud attack or maybe a promotion on that exact date
#but what it is for sure is an anomaly

#since this anomaly affect the clustering we were trying to make
#we take them off and start over
dt.grouped.2 <- dt.grouped[km$cluster-1 != 1]

#we create a distance/dissimilarity matrix
set.seed(100)
gower_dist.2 <- daisy(as.data.frame(dt.grouped.2),
                      metric = "gower")

#silhouette method
k <- 2:7
avg_sil <- sapply(k, silhouette_score, gower_dist.2)
plot(k, type = 'b', avg_sil, xlab = 'Number of clusters', ylab = 'Average Silhouette Scores', frame = FALSE)
#peak 2 (high) and 3 (2nd high)

#elbow method
k <- 1:7
wss_values <- map_dbl(k, wss, gower_dist.2)

plot(k, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
#inflection points at 3 (high)

#just like before, the second highest clusters where k=4, now that we removed 1
#cluster, both of our analysis combined give us k=3

#we proceed to use K=3
km <- kmeans(x = gower_dist.2, centers = 3)
km$size
prop.table(km$size)
#now the clusters are more balanced, probably indicating different profiles

#in order to know with variables are the most important in determining the cluster belonging we train a supervised model
#first we must convert categorical variables into dummies
cols.for.dummy.2 <- c('MODE_MERCHANT', 'MODE_CAC_1', 'MODE_DIRECT', 'MODE_DAY', 'MODE_MONTH')
dt.dummies.2 <- dummy_cols(dt.grouped.2, select_columns = cols.for.dummy.2, remove_selected_columns = TRUE)

#we bind the "target" column which is the cluster (1 and 2) converted to 0 and 1
dt.sup <- cbind(dt.dummies.2, cluster = km$cluster-1)

#we train a classification model, in this case it is a binary classification since we have 2 clusters
#but in order to make it for general purposes (more than 2 clusters)
#we train a multinomial xgboost model

#number of classes
k <- length(km$size)
#definition of the multinomial model
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = k)
#number of iterations
nround <- 500 
#number of folds for the cross validation
cv.nfold <- 5

#we declare the train data matrix
train_matrix <- xgb.DMatrix(data = as.matrix(dt.sup[ , -80]), label = dt.sup$cluster)

#we do a cross validation to get the best iteration parameter
set.seed(5)
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE,
                   early_stopping_rounds = 50)

#we train the xgboost with the best number of iterations
set.seed(10)
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = cv_model$best_iteration)

#we plot the top 10 variables
importance <- xgb.importance(feature_names = colnames(train_matrix), model = bst_model)
xgb.plot.importance(importance_matrix = importance, top_n = 15, measure = 'Frequency')
(imp_features <- head(importance[order(Frequency, decreasing = TRUE)], 15)[ , Feature])

imp_features <- c(imp_features, 'cluster')
#we use these features to see what characterizes the clusters
round(dt.sup[ , ..imp_features][ , lapply(.SD, mean), cluster][order(cluster)], 2)

#cluster 1 
#steady transactions (no outliers), 63% over the median but the lowest avg ticket
#it doesnt have a high amount trx and 77% of trx is on fuel 
#no amount in education, profile of a taxi driver or pizza delivery

#cluster 3
#highest trx ticket but not even 50% transactions over the median
#so he must buy a lot (most num_trx) of little stuff and sometimes big things
#due to the highest max_trx, it has the most outliers and he is making trx
#at least twice as much as cluster 2 and 4 times more than cluster 1 on the 
#weekends, so he must go on spending spree those days with high amt trx
#profile of a wealthy person

#cluster 2
#avg ticket, avg num_trx, he buys diffent varieties of things because of
#mode_cac other with high value and mode_merchant_other with high value, 
#he doesnt use amazon, he doesnt spend on fuel and more than 50% of his trx
#are on schools, middle class student profile