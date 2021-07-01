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

# colnames(mRNA)
Pheno <- read.csv("pheno_train.csv", row.names = 1)
#rename to get disease as an extra column HD, ignores sex, includes age
HD <- sub(Pheno$Name, pattern = "fe", replacement = "")
HD <- sub(HD, pattern = "male_", replacement = "")
HD <- sub(HD, pattern = "Q20", replacement = "WT")
HD <- sub(HD, pattern = "Q111", replacement = "HD")
HD <- sub(HD, pattern = "Q140", replacement = "HD")
HD <- sub(HD, pattern = "Q175", replacement = "HD")
HD <- sub(HD, pattern = "Q80", replacement = "HD")
HD <- sub(HD, pattern = "Q92", replacement = "HD")
#X <- mRNA[apply(mRNA[,-1], 1, function(x) !all(x < 50)),]
#Y <- X[!rowSums(X == 0) >= 20, , drop = FALSE]
#Z <- X[which(rownames(X) %in% rownames(Y) == FALSE),]
m <- mapply(sig_train_mRNA, FUN=as.integer)
rownames(m) <- rownames(sig_train_mRNA)

# create Conditions file
Samples <- colnames(m)
Conditions <- HD
colData <- cbind(Samples, Conditions)
rownames(colData) <- colnames(m)
write.csv(colData, file="individual_disease_train.csv")
print("reached checkpoint")
############################################
# mRNA_ML_data
train_mRNA_t <- transpose(sig_train_mRNA)
rownames(train_mRNA_t) <- colnames(sig_train_mRNA)
colnames(train_mRNA_t) <- rownames(sig_train_mRNA)
t_mRNA_train <- train_mRNA_t %>%
  rownames_to_column(var = "Samples")

t_mRNA_train <- as.data.frame(t_mRNA_train)
colData <- as.data.frame(colData)

# write.csv(t_mRNA_train, file="train_mRNA_t_DELETE LATER.csv")

mRNA_ML_train <- setDT(t_mRNA_train)[setDT(colData), Conditions := i.Conditions, on="Samples"]
column_to_rownames(mRNA_ML_train, "Samples")
write.csv(mRNA_ML_train, file="ML_data_ft_selected")
print("reached next checkpoint")
####################################################################

############################################
# miRNA_ML_data

train_miRNA_t <- transpose(sig_train_miRNA)
rownames(train_miRNA_t) <- colnames(sig_train_miRNA)
colnames(train_miRNA_t) <- rownames(sig_train_miRNA)
t_miRNA_train <- train_miRNA_t %>%
  rownames_to_column(var = "Samples")

t_miRNA_train <- as.data.frame(t_miRNA_train)
colData <- as.data.frame(colData)

# write.csv(t_mRNA_train, file="train_mRNA_t_DELETE LATER.csv")

miRNA_ML_train <- setDT(t_miRNA_train)[setDT(colData), Conditions := i.Conditions, on="Samples"]
miRNA_ML_train <-  column_to_rownames(miRNA_ML_train, "Samples")
write.csv(miRNA_ML_train, file="miRNA_ML_data.csv")


#####################################################################
q()
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\ML_input")
miRNA <- read.csv("../../InputForRFiltering/sig_miRNA_counts.csv", row.names = 1)
colnames(miRNA) <- gsub(colnames(miRNA), pattern = "\\.", replacement = "-")
mRNA <- read.csv("../../InputForRFiltering/sig_mRNA_counts.csv", row.names = 1)

#mRNA <- mRNA[which(rownames(mRNA) %in% rownames(miRNA)),]
#miRNA <- miRNA[which(rownames(miRNA) %in% rownames(mRNA)),]
mRNA$Samples <- NULL

intersect(rownames(mRNA), rownames(miRNA))
# consider if you want to combine the datasets or not, then debug this line if wishing to combine
Data <- cbind(mRNA, miRNA)
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
# input for ML - training data

Data$Samples <- Samples

# I am lazy

Data$Samples <- sub(Data$Samples, pattern = "_10m", replacement = "")
Data$Samples <- sub(Data$Samples, pattern = "_6m", replacement = "")


write.csv(Data, "../../Early Detection/ML_input/ML_Data.csv", row.names = TRUE)

# Validation
miRNA_val <- read.table("../../Early Detection/ML_input/validation_miRNA_counts.txt", row.names = 1)
mRNA_val <- read.table("../../Early Detection/ML_input/validation_mRNA_counts.txt", row.names = 1)

mRNA_val$Samples <- NULL

intersect(rownames(mRNA_val), rownames(miRNA_val))
Data <- cbind(mRNA_val, miRNA_val)
Samples <- Data$Samples
Data$Samples <- NULL

res.pca <- prcomp(Data, scale = TRUE)
fviz_eig(res.pca)
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
# input for ML - validation data
Data$Samples <- Samples
Data$Samples <- sub(Data$Samples, pattern = "_2m", replacement = "")
write.csv(Data, "../../Early Detection/ML_input/ML_Data_val.csv", row.names = TRUE)
print("finished")