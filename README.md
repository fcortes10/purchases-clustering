------------------------------------------------------------------------

> This report is just for academical purposes

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

------------------------------------------------------------------------

#### Data reading

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

------------------------------------------------------------------------

#### Data cleaning

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

------------------------------------------------------------------------

#### Data exploration

For the data exploration we don’t want pretty charts yet, just see how
the data looks. We start with out only numeric column
`ORIGINAL_GROSS_AMT`. To explore a univariate numeric variable, the
simplest way is plotting a histogram:

``` r
hist(dt[ , ORIGINAL_GROSS_AMT])
```

![](README_files/figure-gfm/histogram-1.png)<!-- -->

where we see that we got outliers that doesn’t let us see clearly our
data. So we limit our graph to be between the quantiles 5% and 95%

``` r
ext_q <- quantile(dt[ , ORIGINAL_GROSS_AMT], probs = c(0.05, 0.95))
hist(dt[between(ORIGINAL_GROSS_AMT, ext_q[1], ext_q[2]), ORIGINAL_GROSS_AMT])
```

![](README_files/figure-gfm/quantiles_hist-1.png)<!-- -->

``` r
#exploring number of distinct values in each column
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

``` r
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

``` r
#we see that it is somehow related to institutions or schools but since we got no metadata and the are 88 categories we chose to drop it
dt[ , TRANS_CAC_DESC_2 := NULL]

#we show the 20 most frequent merchant names
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

``` r
#we see that amazon has more than 5 variations in its name, so we group it
dt[grepl('amazon', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'amazon']

#we do the same for other similar cases
dt[grepl('asda', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'asda']
dt[grepl('travelodge', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'travelodge']
dt[grepl('argos', tolower(dt[ , MERCHANT_NAME])), MERCHANT_NAME := 'argos']

#everything else below 500 trx is grouped in a bag
head(dt[ , .N, MERCHANT_NAME][order(N, decreasing = TRUE)], 20)
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
    ## 11:         mrh six ways  283
    ## 12:    rontec longbridge  260
    ## 13:    sainsburys s/mkts  251
    ## 14:      screwfix direct  228
    ## 15:            trainline  225
    ## 16:          park mobile  224
    ## 17:    malthurst limited  218
    ## 18:                 aldi  217
    ## 19:            mcdonalds  205
    ## 20:     sainsburys s/mkt  201

``` r
common.merchants <- c('amazon', 'travelodge', 'asda', 'argos', 'post office counter')
dt[!dt[ , MERCHANT_NAME] %in% common.merchants, MERCHANT_NAME := 'other']

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
