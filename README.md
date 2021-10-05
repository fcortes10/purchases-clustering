------------------------------------------------------------------------

> This report is just for assessment purposes

*by Fernando Cortés Tejada* \|
[linkedin](https://www.linkedin.com/in/fernando-cortes-tejada/) \|
[github](https://github.com/fcortes10)

# Purchase Card Transactions

We have a collection of purchase card transactions for the Birmingham
City Council. This is a historical open source dataset and can be found
in this
[link](https://data.birmingham.gov.uk/dataset/purchase-card-transactions).

The aim of this analysis is to **discover profiles** or **unusual
transactions**. In Data Science language this can be read as clustering
and anomalies detection problems.

The card transactions data starts in April 2014 and ends in January
2018. We want to use the most recent complete year for the analysis so
we chose the whole 2017 year. When looking at the raw data, December
2017 file has different type of data, so the file might be wrong. We
switched to December 2016 to November 2017.

We chose to approach this problem with `R` instead of `python` because
the most robust method for clustering when you have different data
types, e.g. numerical, logical, categorical and ordinal, is Gower’s
distance and is not yet well implemented in a python package. The
mathematical definition of this distance can be found
[here](https://statisticaloddsandends.wordpress.com/2021/02/23/what-is-gowers-distance/).

------------------------------------------------------------------------

## Index

1.  [Data reading](#data-reading)  
2.  [Data cleaning](#data-cleaning)  
3.  [Data exploration](#data-exploration)
4.  [Feature engineering](#feature-engineering)
    -   [Transaction level](#transaction-level)
    -   [Client level](#client-level)  
5.  [Clustering and anomalies
    detection](#clustering-and-anomalies-detection)
    -   [Number of clusters (K)](#number-of-clusters-k)  
    -   [Clustering](#clustering)  
    -   [Anomalies detection](#anomalies-detection)  
    -   [New number of clusters (K)](#new-number-of-clusters-k)  
    -   [New clustering](#new-clustering)  
6.  [Interpretation and conclusions](#interpretation-and-conclusions)

------------------------------------------------------------------------

## Let’s get started

We begin by setting some global configurations and loading required
packages. For installing the ones you don’t have run
`install.packages("package-name")` in the R console.

``` r
Sys.setlocale("LC_TIME", "C")

library(knitr)
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
```

([back to index](#index))

------------------------------------------------------------------------

### Data reading

Then we read the data structure, get all files from the `data-csv`
folder, also
[here](https://github.com/fcortes10/purchases-clustering/tree/main/data-csv),
standardize column names and append all datasets into a big one called
`dt`.

``` r
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

#remove unnecessary/remaining environment variables and cleaning garbage in RAM
rm(list.dt, files, column.names)
gc()
```

Now we can have an overview of how the dataset looks like (just the 3
first rows)

``` r
head(dt, 3)
```

    ##    TRANS_DATE TRANS_VAT_DESC ORIGINAL_GROSS_AMT      MERCHANT_NAME
    ## 1:   22/12/16             VR              20.00    shell kings 587
    ## 2:   15/12/16             VR              35.00    shell kings 587
    ## 3:   22/12/16             VR              75.97 shell fiveways 387
    ##         CARD_NUMBER TRANS_CAC_CODE_1 TRANS_CAC_DESC_1 TRANS_CAC_CODE_2
    ## 1: ************5770             K020     Vehicle Fuel            RV12N
    ## 2: ************5770             K020     Vehicle Fuel            RV12N
    ## 3: ************1147             K020     Vehicle Fuel            RV1K2
    ##           TRANS_CAC_DESC_2 TRANS_CAC_CODE_3         DIRECTORATE
    ## 1:    African-Caribbean DC              A00 Adult & Communities
    ## 2:    African-Caribbean DC              A00 Adult & Communities
    ## 3: Elders Group - Ladywood              A00 Adult & Communities

([back to index](#index))

------------------------------------------------------------------------

### Data cleaning

We apply some treatments to data in order to make it more handleable.
This includes, converting data types, changing formats, dropping
columns, cleaning missing values, etc.

``` r
#data cleaning
#extract just the numeric part from the card number as key (leaving it as string)
dt[ , CARD_NUMBER := str_pad(parse_number(dt[ , CARD_NUMBER]), width = 4, side = 'left', pad = "0")]

#transform the transaction date from string to date format
dt[ , TRANS_DATE := as.Date(dt[ , TRANS_DATE], format = "%d/%m/%y")]

#dropping transaction codes because we are keeping the description
dt[ , c("TRANS_CAC_CODE_1", "TRANS_CAC_CODE_2", "TRANS_CAC_CODE_3") := NULL]

#dropping TRANS_VAT_DESC because there is not metadata and we cannot infer its meaning
dt[ , TRANS_VAT_DESC := NULL]
```

We get a brief summary of the data to see any pattern or issue

``` r
summary(dt)
```

    ##    TRANS_DATE         ORIGINAL_GROSS_AMT MERCHANT_NAME      CARD_NUMBER       
    ##  Min.   :2016-10-25   Length:51592       Length:51592       Length:51592      
    ##  1st Qu.:2017-03-08   Class :character   Class :character   Class :character  
    ##  Median :2017-06-09   Mode  :character   Mode  :character   Mode  :character  
    ##  Mean   :2017-06-07                                                           
    ##  3rd Qu.:2017-09-13                                                           
    ##  Max.   :2017-12-01                                                           
    ##  NA's   :1                                                                    
    ##  TRANS_CAC_DESC_1   TRANS_CAC_DESC_2   DIRECTORATE       
    ##  Length:51592       Length:51592       Length:51592      
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ## 

The first variable we can see is `TRANS_DATE`, which has only one `NA`,
so we remove it.

``` r
dt <- dt[!is.na(TRANS_DATE)]
```

We also see that `ORIGINAL_GROSS_AMT` is a character column when it must
be numeric. We cast it as numeric.

``` r
head(as.numeric(dt[ , ORIGINAL_GROSS_AMT]))
```

    ## Warning in head(as.numeric(dt[, ORIGINAL_GROSS_AMT])): NAs introducidos por
    ## coerción

    ## [1] 20.00 35.00 75.97 50.00 47.24 76.45

where we get a warning of induced `NAs`, so something must be happening.
Checking the `NAs`

``` r
head(dt[which(is.na(as.numeric(dt[ , ORIGINAL_GROSS_AMT])))], 3)
```

    ## Warning in which(is.na(as.numeric(dt[, ORIGINAL_GROSS_AMT]))): NAs introducidos
    ## por coerción

    ##    TRANS_DATE ORIGINAL_GROSS_AMT          MERCHANT_NAME CARD_NUMBER
    ## 1: 2016-12-19          61,206.88 the furnishing service        6583
    ## 2: 2016-12-19          39,520.09 the furnishing service        6583
    ## 3: 2016-12-19          45,585.13 the furnishing service        6583
    ##     TRANS_CAC_DESC_1 TRANS_CAC_DESC_2         DIRECTORATE
    ## 1: Equip Operational      Social Fund Corporate Resources
    ## 2: Equip Operational      Social Fund Corporate Resources
    ## 3: Equip Operational      Social Fund Corporate Resources

we can see it is the thousands separator. So we replace the character
`","` in the string and cast again.

``` r
#it is the thousands separator, we replace it and cast again as numeric
dt[ , ORIGINAL_GROSS_AMT := as.numeric(gsub(",", "", ORIGINAL_GROSS_AMT))]
```

([back to index](#index))

------------------------------------------------------------------------

### Data exploration

For the data exploration we don’t want pretty charts yet, just see how
the data looks. We start with out only numeric column
`ORIGINAL_GROSS_AMT`. To explore a univariate numeric variable, the
simplest way is plotting a histogram:

``` r
hist(dt[ , ORIGINAL_GROSS_AMT], main = "Histogram for gross amount", xlab = "Gross amount")
```

![](README_files/figure-gfm/histogram-1.png)<!-- -->

where we see that we got outliers that doesn’t let us see clearly our
data. So we limit our graph to be between the quantiles 5% and 95%

``` r
ext_q <- quantile(dt[ , ORIGINAL_GROSS_AMT], probs = c(0.05, 0.95))
hist(dt[between(ORIGINAL_GROSS_AMT, ext_q[1], ext_q[2]), ORIGINAL_GROSS_AMT], 
     main = "Histogram for gross amount (without tail values)", xlab = "Gross amount")
```

![](README_files/figure-gfm/quantiles_hist-1.png)<!-- -->

We can see a right skewed distribution, similar to a decaying
exponential.

Now we will explore the categorical columns by checking the number of
distinct values in each variable.

``` r
#we declare a function for unique values
f <- function(x){
  length(unique(x))
}

#we apply the function to the margin 2 (columns)
apply(dt, MARGIN = 2, f)
```

    ##         TRANS_DATE ORIGINAL_GROSS_AMT      MERCHANT_NAME        CARD_NUMBER 
    ##                370              18083               6268               1028 
    ##   TRANS_CAC_DESC_1   TRANS_CAC_DESC_2        DIRECTORATE 
    ##                125                888                 13

We see that we have:

-   `370` distinct days with transactions
-   `18,083` distinct monetary amounts in transactions
-   `6,268` distinct merchant
-   `1,028` distinct card numbers (and we can assume `1,028` distinct
    clients)
-   `125` distinct type of business according to `DESC_1`
-   `888` distinct type of business according to `DESC_2`
-   `13` distinct type of business according to `DIRECTORATE`

We start with `TRANS_CAC_DESC_2` and show the 20 most frequent
categories:

``` r
head(dt[ , .N, TRANS_CAC_DESC_2][order(N, decreasing = TRUE)], 20)
```

    ##                             TRANS_CAC_DESC_2    N
    ##  1:            Homeless Private Sector Accom 4987
    ##  2:   Illegal Money Lending T Stds Comm Inv. 2957
    ##  3:            The City of Birmingham School 1386
    ##  4:                      Camborne House HLDC  871
    ##  5:                           Technical Unit  779
    ##  6:                       West Heath Primary  564
    ##  7:     St Barnabas CE Junior & Infant  (NC)  510
    ##  8:                               Bloomsbury  472
    ##  9:           Ward End Junior & Infant  (NC)  462
    ## 10:                              Baskerville  453
    ## 11:              16+ Accommodation & Support  409
    ## 12:     Warwick Hse HLDC, 938 Warwick Rd B27  402
    ## 13:        Regents Park Junior & Infant (NC)  396
    ## 14:                      Harper Bell Primary  393
    ## 15:                 Selly Oak Nursery School  379
    ## 16:              St Edmund Campion Secondary  336
    ## 17:          Raddlebarn Junior & Infant (NC)  330
    ## 18:                        Corp Inbound Post  311
    ## 19:                                 Uffculme  309
    ## 20: International School & Community College  306

where we can see that it is somehow related to institutions or schools
but since we got no metadata and the are 88 categories we chose to drop
it.

``` r
dt[ , TRANS_CAC_DESC_2 := NULL]
```

Then, we continue with the 20 most frequent `MERCHANT_NAME`

``` r
head(dt[ , .N, MERCHANT_NAME][order(N, decreasing = TRUE)], 20)
```

    ##                 MERCHANT_NAME    N
    ##  1:     amazon uk marketplace 5153
    ##  2:         travelodge gb0000 4890
    ##  3:          amazon uk retail 1388
    ##  4: amazon uk retail amazon.c 1038
    ##  5:        asda home shopping  962
    ##  6: amazon svcs eu-uk amazon.  641
    ##  7:              amazon.co.uk  627
    ##  8:       post office counter  585
    ##  9:           asda superstore  456
    ## 10:        argos retail group  380
    ## 11:      rontec weoley castle  367
    ## 12:         js online grocery  364
    ## 13:        bcc register offic  357
    ## 14:          texaco ash motor  308
    ## 15:           www.argos.co.uk  307
    ## 16:         esso aston way ss  305
    ## 17:              mrh six ways  283
    ## 18:         rontec longbridge  260
    ## 19:         sainsburys s/mkts  251
    ## 20:    amazon eu amazon.co.uk  244

where we see that amazon has more than 5 variations in its name, so we
group it

``` r
dt[grepl('amazon', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'amazon']
```

and we do the same for other similar cases.

``` r
dt[grepl('asda', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'asda']
dt[grepl('travelodge', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'travelodge']
dt[grepl('argos', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'argos']
```

We just keep the common merchants and everything else is grouped in a
bag to reduce categories.

``` r
head(dt[ , .N, MERCHANT_NAME][order(N, decreasing = TRUE)], 10)
```

    ##            MERCHANT_NAME    N
    ##  1:               amazon 9421
    ##  2:           travelodge 4990
    ##  3:                 asda 1736
    ##  4:                argos  961
    ##  5:  post office counter  585
    ##  6: rontec weoley castle  367
    ##  7:    js online grocery  364
    ##  8:   bcc register offic  357
    ##  9:     texaco ash motor  308
    ## 10:    esso aston way ss  305

``` r
common.merchants <- c('amazon', 'travelodge', 'asda', 'argos', 'post office counter')
dt[!dt[ , MERCHANT_NAME] %in% common.merchants, MERCHANT_NAME := 'other']
```

We do the same for `TRANS_CAC_DESC_1` and `DIRECTORATE` but we define a
threshold of 5% not to put the category in a bag.

``` r
#we show the 20 most frequent desc 1
head(dt[ , .N, TRANS_CAC_DESC_1][order(N, decreasing = TRUE)], 20)
```

    ##            TRANS_CAC_DESC_1     N
    ##  1:       Equip Operational 10209
    ##  2:            Vehicle Fuel  5560
    ##  3:          Purchases Food  5006
    ##  4:     Other Third Parties  4926
    ##  5:      Supplies & Sev Mic  2799
    ##  6:                   Books  2555
    ##  7:          Mat'l Raw/Drct  1720
    ##  8: Conference Fees Subs UK  1549
    ##  9:             Equip Other  1481
    ## 10:    Bldg RM Departmental  1078
    ## 11:              Stationery  1047
    ## 12:             Hospitality  1023
    ## 13:         Prof Fees other   989
    ## 14:         Travel Bus/Rail   842
    ## 15:                 Postage   797
    ## 16:    Phon NonCentrx Lines   720
    ## 17:         Computing Other   674
    ## 18:             Electricity   645
    ## 19:      Other Fix&Fittings   623
    ## 20:     Vehicle OthrunCosts   564

``` r
#we keep the groups with more than 5% of total transactions and the rest is grouped in a bag
(gt5pct <- dt[ , .N, TRANS_CAC_DESC_1][order(N, decreasing = TRUE)][N > 0.05*nrow(dt), ][ , TRANS_CAC_DESC_1])
```

    ## [1] "Equip Operational"   "Vehicle Fuel"        "Purchases Food"     
    ## [4] "Other Third Parties" "Supplies & Sev Mic"

``` r
dt[!dt[ , TRANS_CAC_DESC_1] %in% gt5pct, TRANS_CAC_DESC_1 := 'other']
```

``` r
#we show the 10 most frequent directorate
dt[ , DIRECTORATE := toupper(DIRECTORATE)]
head(dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)], 10)
```

    ##                      DIRECTORATE     N
    ##  1:                CYP&F SCHOOLS 24139
    ##  2:               LOCAL SERVICES 14162
    ##  3:                        CYP&F  5607
    ##  4:          CORPORATE RESOURCES  3962
    ##  5:          ADULT & COMMUNITIES  1942
    ##  6:                         #N/D   738
    ##  7:                  DEVELOPMENT   663
    ##  8: ADULT SOCIAL CARE AND HEALTH   368
    ##  9:                       ADULTS     6
    ## 10:                        CYO&F     4

``` r
#we keep the groups with more than 5% of total transactions and the rest is grouped in a bag
(gt5pct <- dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)][N > 0.05*nrow(dt), ][ , DIRECTORATE])
```

    ## [1] "CYP&F SCHOOLS"       "LOCAL SERVICES"      "CYP&F"              
    ## [4] "CORPORATE RESOURCES"

``` r
dt[!dt[ , DIRECTORATE] %in% gt5pct, DIRECTORATE := 'other']
head(dt[ , .N, DIRECTORATE][order(N, decreasing = TRUE)], 10)
```

    ##            DIRECTORATE     N
    ## 1:       CYP&F SCHOOLS 24139
    ## 2:      LOCAL SERVICES 14162
    ## 3:               CYP&F  5607
    ## 4: CORPORATE RESOURCES  3962
    ## 5:               other  3721

([back to index](#index))

------------------------------------------------------------------------

### Feature engineering

We have divided the feature engineering in two groups: transaction level
and client level.

#### Transaction level

We just have 6 columns and one is the key column (card\_number) so just
5 features. Thus, we need to create more features in order to make
clusters and find profiles.

Let’s begin by extracting the day, weekday and month as variables.

``` r
#extract the day as a variable
dt[ , DAY := as.numeric(substr(x = TRANS_DATE, start = 9, stop = 10))]

#extract the weekday as a variable
dt[ , WEEKDAY := weekdays(dt[ , TRANS_DATE])]

#extract the months as a variable
dt[ , MONTH := as.numeric(substr(x = TRANS_DATE, start = 6, stop = 7))]
```

We create the `CHARGEBACK` feature, which tells us if the transaction
amount is negative (a return)

``` r
#chargebacks
dt[ , CHARGEBACK := ifelse(ORIGINAL_GROSS_AMT < 0, 1, 0)]
```

and with that we change all amounts to positive.

``` r
#amounts to positive
dt[ , POSITIVE_AMT := ifelse(CHARGEBACK == 1, -1*ORIGINAL_GROSS_AMT, ORIGINAL_GROSS_AMT)]
```

We also create binary features that indicate us if the transaction
amount is an outlier, an extreme value, is a tail value or is over the
median.

``` r
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
```

Now we create two more binary features related to transactions made
around the payday or on weekends.

``` r
#transactions around payday
paydays <- c(1, 2, 14, 15, 16, 28, 29, 30, 31)
dt[ , PAYDAY_TRX := ifelse(DAY %in% paydays, 1, 0)]

#transactions on weekends
weekend.days <- c('Saturday', 'Sunday')
dt[ , WEEKEND_TRX := ifelse(WEEKDAY %in% weekend.days, 1, 0)]
```

We see a summary with the new features.

``` r
summary(dt)
```

    ##    TRANS_DATE         ORIGINAL_GROSS_AMT  MERCHANT_NAME      CARD_NUMBER       
    ##  Min.   :2016-10-25   Min.   :-486980.1   Length:51591       Length:51591      
    ##  1st Qu.:2017-03-08   1st Qu.:     13.2   Class :character   Class :character  
    ##  Median :2017-06-09   Median :     42.9   Mode  :character   Mode  :character  
    ##  Mean   :2017-06-07   Mean   :    212.7                                        
    ##  3rd Qu.:2017-09-13   3rd Qu.:    104.8                                        
    ##  Max.   :2017-12-01   Max.   : 280102.2                                        
    ##  TRANS_CAC_DESC_1   DIRECTORATE             DAY          WEEKDAY         
    ##  Length:51591       Length:51591       Min.   : 1.00   Length:51591      
    ##  Class :character   Class :character   1st Qu.: 8.00   Class :character  
    ##  Mode  :character   Mode  :character   Median :15.00   Mode  :character  
    ##                                        Mean   :15.31                     
    ##                                        3rd Qu.:22.00                     
    ##                                        Max.   :31.00                     
    ##      MONTH         CHARGEBACK       POSITIVE_AMT         OUTLIER      
    ##  Min.   : 1.00   Min.   :0.00000   Min.   :     0.0   Min.   :0.0000  
    ##  1st Qu.: 3.00   1st Qu.:0.00000   1st Qu.:    15.0   1st Qu.:0.0000  
    ##  Median : 6.00   Median :0.00000   Median :    47.0   Median :0.0000  
    ##  Mean   : 6.43   Mean   :0.03923   Mean   :   265.7   Mean   :0.1419  
    ##  3rd Qu.:10.00   3rd Qu.:0.00000   3rd Qu.:   118.0   3rd Qu.:0.0000  
    ##  Max.   :12.00   Max.   :1.00000   Max.   :486980.1   Max.   :1.0000  
    ##  EXTREME_VALUE      TAIL_VALUE           OTM           PAYDAY_TRX    
    ##  Min.   :0.0000   Min.   :0.00000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.0000  
    ##  Median :0.0000   Median :0.00000   Median :0.0000   Median :0.0000  
    ##  Mean   :0.1084   Mean   :0.04987   Mean   :0.4994   Mean   :0.2407  
    ##  3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:1.0000   3rd Qu.:0.0000  
    ##  Max.   :1.0000   Max.   :1.00000   Max.   :1.0000   Max.   :1.0000  
    ##   WEEKEND_TRX     
    ##  Min.   :0.00000  
    ##  1st Qu.:0.00000  
    ##  Median :0.00000  
    ##  Mean   :0.05956  
    ##  3rd Qu.:0.00000  
    ##  Max.   :1.00000

([back to index](#index))

#### Client level

Based on what we have just engineered at the transaction level, we begin
to create features by grouping the information by client.

First, we need to create dummies from the categorical variables.

``` r
#select columns to make dummies
cols.for.dummy <- c('MERCHANT_NAME', 'TRANS_CAC_DESC_1', 'DIRECTORATE', 'WEEKDAY')
dt.dummies <- dummy_cols(dt, select_columns = cols.for.dummy)
```

Then we start grouping by client and creating new features. These new
features include averages, totals, sums, maximums, percentages, ratios
and modes.

``` r
#declare the statistical mode function (since R doesn't have one)
getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

#group by client
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
```

Finally, we see a summary of the new client-level features.

``` r
summary(dt.grouped)
```

    ##  CARD_NUMBER           NUM_TRX           AVG_TRX            MAX_TRX        
    ##  Length:1028        Min.   :   1.00   Min.   :     0.8   Min.   :     0.8  
    ##  Class :character   1st Qu.:   8.00   1st Qu.:    47.0   1st Qu.:    90.0  
    ##  Mode  :character   Median :  26.00   Median :    66.9   Median :   276.0  
    ##                     Mean   :  50.19   Mean   :   752.0   Mean   :  1584.6  
    ##                     3rd Qu.:  58.25   3rd Qu.:   101.0   3rd Qu.:   500.0  
    ##                     Max.   :1486.00   Max.   :486980.1   Max.   :486980.1  
    ##                                                                            
    ##  NUM_CHARGEBACKS   PCT_CHARGEBACKS   AVG_AMT_CHARGEBACKS PCT_AMT_CHARGEBACKS
    ##  Min.   :  0.000   Min.   :0.00000   Min.   :     0.0    Min.   :0.000000   
    ##  1st Qu.:  0.000   1st Qu.:0.00000   1st Qu.:     0.0    1st Qu.:0.000000   
    ##  Median :  0.000   Median :0.00000   Median :     0.0    Median :0.000000   
    ##  Mean   :  1.969   Mean   :0.05280   Mean   :   637.0    Mean   :0.049113   
    ##  3rd Qu.:  1.000   3rd Qu.:0.02862   3rd Qu.:     0.8    3rd Qu.:0.009714   
    ##  Max.   :234.000   Max.   :1.00000   Max.   :486980.1    Max.   :1.000000   
    ##                                                                             
    ##   NUM_OUTLIER        PCT_OUTLIER       NUM_XTRM_VALUE     PCT_XTRM_VALUE   
    ##  Min.   :   0.000   Min.   :0.000000   Min.   :   0.000   Min.   :0.00000  
    ##  1st Qu.:   0.000   1st Qu.:0.000000   1st Qu.:   0.000   1st Qu.:0.00000  
    ##  Median :   1.000   Median :0.009489   Median :   0.000   Median :0.00000  
    ##  Mean   :   7.121   Mean   :0.089226   Mean   :   5.438   Mean   :0.05314  
    ##  3rd Qu.:   3.000   3rd Qu.:0.075758   3rd Qu.:   1.000   3rd Qu.:0.02332  
    ##  Max.   :1348.000   Max.   :1.000000   Max.   :1271.000   Max.   :1.00000  
    ##                                                                            
    ##  NUM_TAIL_VALUE    PCT_TAIL_VALUE       NUM_OTM           PCT_OTM      
    ##  Min.   :  0.000   Min.   :0.00000   Min.   :   0.00   Min.   :0.0000  
    ##  1st Qu.:  0.000   1st Qu.:0.00000   1st Qu.:   3.00   1st Qu.:0.3058  
    ##  Median :  0.000   Median :0.00000   Median :  12.00   Median :0.4791  
    ##  Mean   :  2.503   Mean   :0.03498   Mean   :  25.06   Mean   :0.5120  
    ##  3rd Qu.:  1.000   3rd Qu.:0.02709   3rd Qu.:  27.00   3rd Qu.:0.7500  
    ##  Max.   :305.000   Max.   :1.00000   Max.   :1485.00   Max.   :1.0000  
    ##                                                                        
    ##  NUM_PAYDAY_TRX   PCT_PAYDAY_TRX   NUM_WEEKEND_TRX   PCT_WEEKEND_TRX  
    ##  Min.   :  0.00   Min.   :0.0000   Min.   :  0.000   Min.   :0.00000  
    ##  1st Qu.:  2.00   1st Qu.:0.1429   1st Qu.:  0.000   1st Qu.:0.00000  
    ##  Median :  6.00   Median :0.2258   Median :  0.000   Median :0.00000  
    ##  Mean   : 12.08   Mean   :0.2303   Mean   :  2.989   Mean   :0.06267  
    ##  3rd Qu.: 14.00   3rd Qu.:0.3000   3rd Qu.:  3.000   3rd Qu.:0.08000  
    ##  Max.   :407.00   Max.   :1.0000   Max.   :108.000   Max.   :1.00000  
    ##                                                                       
    ##              MODE_MERCHANT               MODE_CAC_1               MODE_DIRECT 
    ##  other              :893   Vehicle Fuel       :216   other              :195  
    ##  amazon             : 97   Equip Operational  :155   CYP&F SCHOOLS      :385  
    ##  argos              :  8   Supplies & Sev Mic : 73   CYP&F              :128  
    ##  post office counter:  9   other              :506   CORPORATE RESOURCES: 76  
    ##  asda               : 10   Purchases Food     : 69   LOCAL SERVICES     :244  
    ##  travelodge         : 11   Other Third Parties:  9                            
    ##                                                                               
    ##     MODE_DAY     MODE_MONTH 
    ##  10     : 71   5      :164  
    ##  6      : 70   1      :138  
    ##  9      : 51   11     :138  
    ##  14     : 48   3      : 94  
    ##  20     : 46   7      : 94  
    ##  13     : 46   12     : 87  
    ##  (Other):696   (Other):313

([back to index](#index))

------------------------------------------------------------------------

### Clustering and anomalies detection

#### Number of clusters (K)

([back to index](#index))

#### Clustering

([back to index](#index))

#### Anomalies detection

([back to index](#index))

#### New number of clusters (K)

([back to index](#index))

#### New clustering

([back to index](#index))

------------------------------------------------------------------------

### Interpretation and conclusions

([back to index](#index))
