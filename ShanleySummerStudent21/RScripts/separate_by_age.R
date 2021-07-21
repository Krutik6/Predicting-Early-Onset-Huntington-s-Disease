library(tidyverse)
library(testthat)

# todo continue analysis with and without outliers
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\NormalisedData")
# import data

miRNA <- read.csv("normalized_miRNA_counts.txt", row.names = 1, sep = "\t")
mRNA <- read.csv("normalized_mRNA_counts.txt", row.names = 1, sep="\t")


############################################################
# remove known outliers and mRNA-microRNA inconsistencies
miRNA <- miRNA[,-c(63, 96, 127, 145, 157)]
mRNA <- mRNA[,-c(63, 96, 127, 145, 157)]
#########################################################
# Separate 2m and 6,10m data
mRNA_2m <- mRNA %>% select(contains("2m"))
miRNA_2m <- miRNA %>% select(-contains("2m"))

mRNA_6m <- mRNA %>% select(contains("6m"))
miRNA_6m <- miRNA %>% select(-contains("6m"))

mRNA_10m <- mRNA %>% select(contains("10m"))
miRNA_10m <- miRNA %>% select(contains("10m"))

#########################################################

# save files

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\normalized_age\\")

write.csv(mRNA_2m, "mRNA_2m.csv")
write.csv(miRNA_2m,"miRNA_2m.csv")
write.csv(mRNA_6m, "mRNA_6m.csv")
write.csv(miRNA_6m, "miRNA_6m.csv")
write.csv(mRNA_10m, "mRNA_10m.csv")
write.csv(miRNA_10m, "miRNA_10m.csv")

print("completed separation by age")

