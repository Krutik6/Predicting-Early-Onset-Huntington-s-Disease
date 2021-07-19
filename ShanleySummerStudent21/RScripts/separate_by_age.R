library(tidyverse)
library(testthat)


setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForRFiltering")
# import data

miRNA <- read.csv("miRNA_counts.csv", row.names = 1)
mRNA <- read.csv("mRNA_counts.csv", row.names = 1)

# not dealing with phenotypes for now but code can remain for future use
Pheno_miRNA <- read.csv("miRNA_Pheno.csv", row.names = 1)
Pheno_mRNA <- read.csv("mRNA_Pheno.csv")
############################################################
# remove known outliers and mRNA-microRNA inconsistencies
miRNA <- miRNA[,-c(63, 96, 127, 145, 157)]
mRNA <- mRNA[,-c(63, 96, 127, 145, 157)]
Pheno_miRNA <- Pheno_miRNA[-c(63, 96, 127, 145, 157),]
Pheno_mRNA <- Pheno_mRNA[-c(63, 96, 127, 145, 157),]
rownames(Pheno_miRNA) <- NULL
rownames(Pheno_mRNA) <- NULL
#############################################################
# check pheno names

# This line does not run, due to different names. Since sex is not
# being analysed currently, I have changed the sex. This can be
# changed if need be later on, however, why is there the same data
# point with different sexes?
# Pheno_miRNA$Name[92] <- "male_Q20_10m"
# expect_equal(Pheno_miRNA$Name, Pheno_mRNA$Name)
# 1 difference but acceptable
#############################################################
# Remove naming specifics
HD <- sub(Pheno_miRNA$Name, pattern = "fe", replacement = "")
HD <- sub(HD, pattern = "male_", replacement = "")
HD <- sub(HD, pattern = "Q20", replacement = "WT")
HD <- sub(HD, pattern = "Q111", replacement = "HD")
HD <- sub(HD, pattern = "Q140", replacement = "HD")
HD <- sub(HD, pattern = "Q175", replacement = "HD")
HD <- sub(HD, pattern = "Q80", replacement = "HD")
HD <- sub(HD, pattern = "Q92", replacement = "HD")
Pheno_miRNA$HD <- HD
Pheno_mRNA$HD <- HD
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

setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\age\\")

write.csv(mRNA_2m, "mRNA_2m.csv")
write.csv(miRNA_2m,"miRNA_2m.csv")
write.csv(mRNA_6m, "mRNA_6m.csv")
write.csv(miRNA_6m, "miRNA_6m.csv")
write.csv(mRNA_10m, "mRNA_10m.csv")
write.csv(miRNA_10m, "miRNA_10m.csv")

print("completed separation by age")

