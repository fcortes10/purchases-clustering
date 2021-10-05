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
