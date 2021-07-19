# Title     : Prepare data for ML
# Objective : Transforms the data ready for processing by the classifier

# currently this also combines mRNA, and miRNA data
# code adapted from the archived "Combine_data.R" script
# Created by: Colleen
# Created on: 30/06/2021
library(data.table)
library(factoextra)
library(dplyr)
library(tibble)

############################################
# code extracted from DE_ML_Input_mRNA.R

#set working dir
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Separated_Data\\")


getColnames <- function(RNA_df){
  #rename to get disease as an extra column HD, ignores sex, includes age
  HD <- sub(colnames(RNA_df), pattern = "fe", replacement = "")
  HD <- sub(HD, pattern = "male_", replacement = "")
  HD <- sub(HD, pattern = "_2m", replacement = "")
  HD <- sub(HD, pattern = "_6m", replacement = "")
  HD <- sub(HD, pattern = "_10m", replacement = "")
  HD <- sub(HD, pattern = "Q20", replacement = "WT")
  HD <- sub(HD, pattern = "Q111", replacement = "HD")
  HD <- sub(HD, pattern = "Q140", replacement = "HD")
  HD <- sub(HD, pattern = "Q175", replacement = "HD")
  HD <- sub(HD, pattern = "Q80", replacement = "HD")
  HD <- sub(HD, pattern = "Q92", replacement = "HD")

  #X <- mRNA[apply(mRNA[,-1], 1, function(x) !all(x < 50)),]
  #Y <- X[!rowSums(X == 0) >= 20, , drop = FALSE]
  #Z <- X[which(rownames(X) %in% rownames(Y) == FALSE),]

  m <- mapply(RNA_df, FUN=as.integer)
  rownames(m) <- rownames(RNA_df)

  # create file linking individual to phenotype
  Samples <- colnames(m)
  Conditions <- HD
  colData <- cbind(Samples, Conditions)
  rownames(colData) <- colnames(m)
  colData <- as.data.frame(colData)

  if (any(is.na(colData))){
    stop("Some conditions have NA instead of phenotype. Check that all columns in the df have been parsed")
  }
  return (colData)
}


#########################################################
transform_for_ml <- function(RNA_data, file_name, colData)
{

  RNA_t <- transpose(RNA_data)
  #write.csv(RNA_t, "transposed_deleteme")
  rownames(RNA_t) <- colnames(RNA_data)
  colnames(RNA_t) <- rownames(RNA_data)

  write.csv(RNA_t, "../../Early Detection/Data/Separated_Data/RNA_t.csv")

  t_RNA <- RNA_t %>%
    rownames_to_column(var = "Samples")

  t_RNA <- as.data.frame(t_RNA)

  # write.csv(t_mRNA_train, file="train_mRNA_t_DELETE LATER.csv")
  RNA_ML <- setDT(t_RNA)[setDT(colData), Conditions := i.Conditions, on="Samples"]
  RNA_ML <-  column_to_rownames(RNA_ML, "Samples")
  loc <-"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForML\\"
  f <- paste(loc,file_name,".csv", sep="")
  # check no NAs exist in the dataframe
  if (any(is.na(RNA_ML))){
    stop("NA exists in dataframe:",file_name ,". Check that all columns in the df have been parsed in the get columns function")
  }

  # todo all rows which are equal to 0 should be removed
  #if (any(base::rowSums(RNA_ML) == 0)){
    #warning("Rows containing  all 0s have been found in dataframe. Removing")

    #keep <- rowSums(counts(RNA_ML)) >= 0
    #RNA_ML <- df[keep,]
  #}
  write.csv(RNA_ML, file=f)
  print(paste("saved ", f))

}

s_train_mRNA <- read.csv("../../Early Detection/Data/Separated_Data/mRNA_train.csv", row.names = 1)
s_train_miRNA <- read.csv("../../Early Detection/Data/Separated_Data/miRNA_train.csv", row.names = 1)
s_validate_mRNA <- read.csv("../../Early Detection/Data/Separated_Data/mRNA_validation.csv", row.names = 1)
s_validate_miRNA <- read.csv("../../Early Detection/Data/Separated_Data/miRNA_validation.csv", row.names = 1)

RNAs <- list(s_train_mRNA, s_train_miRNA, s_validate_mRNA, s_validate_miRNA)
file_names <- c("ML_data_train_mRNA", "ML_data_train_miRNA", "ML_data_validate_mRNA", "ML_data_validate_miRNA")

train_col_mRNA <- getColnames(s_train_mRNA)
train_col_miRNA <- getColnames(s_train_miRNA)
val_cols_mRNA <- getColnames(s_validate_mRNA)
val_cols_miRNA <- getColnames(s_validate_miRNA)

cols <- list(train_col_mRNA, train_col_miRNA, val_cols_mRNA, val_cols_miRNA)

i <- 0
for (rna in RNAs){
  i <- i+1
  col <- cols[i]
  transform_for_ml(rna, file_names[i], col[[1]])
}


print("finished")


# Transform RNAs into a form ready for ML
transpose_df <- function(df){
  # first remember the names
  n <- df$name

  # transpose all but the first column (name)
  df <- as.data.frame(t(df[,-1]))
  colnames(df) <- n
  df$myfactor <- factor(row.names(df))

  str(df) # Check the column types
  return (df)
}