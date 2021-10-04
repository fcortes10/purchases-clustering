#global configurations
Sys.setlocale("LC_TIME", "C")

#packages
library(data.table)
library(plyr)
library(readr)
library(stringr)
library(dplyr)
library(ggQC)

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


##feature engineering
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




