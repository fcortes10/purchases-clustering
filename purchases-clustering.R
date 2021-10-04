library(data.table)
library(plyr)

dt <- fread('data-csv/purchase-card-transactions-201612.csv', nrows = 0)

files <- list.files('data-csv', pattern = '..csv', full.names = TRUE)

column.names <- toupper(gsub(" ", "_", colnames(dt)))

list.dt <- lapply(files, fread, select = colnames(dt), col.names = column.names)

dt <- as.data.table(ldply(list.dt, data.frame))

rm(list.dt, files, column.names)
gc()
