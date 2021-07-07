# Title     : Prepare data for ML
# Objective : Transforms the data ready for processing by the classifier
# todo decide if combining mRNA, miRNA data or keeping separate
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
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data")

#load data
sig_train_mRNA <- read.csv("sig_mRNA_train.csv", row.names = 1)
sig_val_mRNA <- read.csv("sig_mRNA_validate.csv", row.names = 1)


getColnames <- function(file_name, RNA_df){
  Pheno <- read.csv(file_name, row.names = 1)
  #rename to get disease as an extra column HD, ignores sex, includes age
  #todo remove the repeats _1R - _4R
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


############################################
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

transform_for_ml <- function(RNA_data, file_name, colData)
{

  RNA_t <- transpose(RNA_data)
  #write.csv(RNA_t, "transposed_deleteme")
  rownames(RNA_t) <- colnames(RNA_data)
  colnames(RNA_t) <- rownames(RNA_data)

  write.csv(RNA_t, "RNA_t.csv")

  t_RNA <- RNA_t %>%
    rownames_to_column(var = "Samples")

  t_RNA <- as.data.frame(t_RNA)


  # write.csv(t_mRNA_train, file="train_mRNA_t_DELETE LATER.csv")

  RNA_ML <- setDT(t_RNA)[setDT(colData), Conditions := i.Conditions, on="Samples"]
  RNA_ML <-  column_to_rownames(RNA_ML, "Samples")


  loc <-"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForML\\"

  f <- paste(loc,file_name,".csv", sep="")
  write.csv(RNA_ML, file=f)
  print(paste("saved ", f))

}

s_train_mRNA <- read.csv("sig_mRNA_train.csv", row.names = 1)
s_train_miRNA <- read.csv("sig_miRNA_train.csv", row.names = 1)
s_validate_mRNA <- read.csv("sig_mRNA_validate.csv", row.names = 1)
s_validate_miRNA <- read.csv("sig_miRNA_validate.csv", row.names = 1)

RNAs <- list(s_train_mRNA, s_train_miRNA, s_validate_mRNA, s_validate_miRNA)
file_names <- c("ML_data_train_mRNA", "ML_data_train_miRNA", "ML_data_validate_mRNA", "ML_data_validate_miRNA")

train_col <- getColnames("pheno_train.csv", s_train_mRNA)
val_cols <- getColnames("pheno_validation.csv", s_validate_mRNA)
cols <- list(train_col, train_col, val_cols, val_cols)

i <- 0
for (rna in RNAs){
  i <- i+1
  col <- cols[i]
  transform_for_ml(rna, file_names[i], col[[1]])
}

#####################################################################

#intersect(rownames(mRNA), rownames(miRNA))
# consider if you want to combine the datasets or not, then debug this line if wishing to combine
# Data <- cbind(mRNA, miRNA)
plot_pca <- function(Data){
  Samples <- Data$Samples
  Data$Samples <- NULL

  res.pca <- prcomp(Data, scale = TRUE)
  fviz_eig(res.pca)
  # if recieve error "Viewport has zero dimension(s)" increase width of plot window
  fviz_pca_ind(res.pca,
               col.ind = "cos2",
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
               repel = TRUE
  )
  fviz_pca_var(res.pca,
               col.var = "contrib", # Color by contributions to the PC
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
               repel = TRUE     # Avoid text overlapping
  )
  fviz_pca_biplot(res.pca, repel = TRUE,
                  col.var = "#2E9FDF", # Variables color
                  col.ind = "#696969"  # Individuals color
  )
  print("i executed")
  }

i <- 0
for (rna in RNAs){
  # todo plot pca should produce figures -> does the input need to be coupled?
  i <- i+1
  plot_pca(rna)
}

print("finished")