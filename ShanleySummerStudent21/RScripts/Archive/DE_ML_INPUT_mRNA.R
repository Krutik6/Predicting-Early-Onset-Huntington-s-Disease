#load libraries
library(DESeq2)
library(limma)
library(dplyr)
library(tidyverse) 
library(factoextra)
#set working dir
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data")
#load data
mRNA <- read.csv("mRNA_train.csv", row.names = 1)
# colnames(mRNA)
Pheno <- read.csv("pheno_train.csv", row.names = 1)
#rename
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
m <- mapply(mRNA, FUN=as.integer)
rownames(m) <- rownames(mRNA)
# create Conditions file
Samples <- colnames(m)
Conditions <- HD
colData <- cbind(Samples, Conditions)
rownames(colData) <- colnames(m)
dds <- DESeqDataSetFromMatrix(countData = m, colData = colData, 
                              design = ~Conditions)
# remove low level genes -- 50 because many samples
keep <- rowSums(counts(dds)) >= 50
dds <- dds[keep,]
#check
plotMDS(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))
boxplot(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))
### normalised counts for ML
dds <- estimateSizeFactors(dds)
sizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)
write.table(normalized_counts, file="normalized_mRNA_counts.txt", sep="\t",
            quote=F, col.names=NA)
###
dds$Conditions <- factor(dds$Conditions, levels = unique(dds$Conditions))
dds$Conditions
dds <- DESeq(dds)
resultsNames(dds)
# DE
getDE <- function(numC, denC){
    res <- results(dds, contrast= c("Conditions", numC, denC))
    res_B <- suppressMessages(as.data.frame(lfcShrink(dds=dds, 
                                                      contrast=c("Conditions",
                                                                 numC,
                                                                 denC), 
                                                      res=res,
                                                      type = 'ashr')))
    return(res_B)
}
M6 <- getDE(numC = 'HD_6m', denC = 'WT_6m')
M10 <- getDE(numC = 'HD_10m', denC = 'WT_10m')
################################################################################
# which genes sig DE in both 6m and 10m
EmptyList <- list()
EmptyList[["M6"]] <- M6
EmptyList[["M10"]] <- M10

SigList <- lapply(EmptyList, function(x) {x[which(x$padj <0.05),]})

NamedList <- lapply(SigList, function(x) {cbind(x, rownames(x))})

cNames <- colnames(NamedList[[1]])
cNames[6] <- "Names"

NamesinList <- lapply(NamedList, setNames, cNames)
colnames(NamesinList[[1]])

Sig_genes <- bind_rows(NamesinList) %>% 
    group_by(Names) %>% 
    summarize(occurance = n()) %>% 
    filter(occurance > 1)
################################################################################
norm_mRNA <- read.table("normalized_mRNA_counts.txt", row.names = 1)
colnames(norm_mRNA) <- colnames(mRNA)
mRNA_consistent <- norm_mRNA[which(rownames(norm_mRNA) %in% Sig_genes$Names),] 

#PCA plots
Data <- t(mRNA_consistent)
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

DF <- as.data.frame(Data)
DF$Samples <- HD
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\ML_input")

write.csv(DF, "../../Early Detection/ML_input/mRNA_data.csv")

################################################################################
#validation data
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data")
mRNA_v <- read.csv("mRNA_validation.csv", row.names = 1)
pheno_v <- read.csv("pheno_validation.csv", row.names = 1)

HD <- sub(pheno_v$Name, pattern = "fe", replacement = "")
HD <- sub(HD, pattern = "male_", replacement = "")
HD <- sub(HD, pattern = "Q20", replacement = "WT")
HD <- sub(HD, pattern = "Q111", replacement = "HD")
HD <- sub(HD, pattern = "Q140", replacement = "HD")
HD <- sub(HD, pattern = "Q175", replacement = "HD")
HD <- sub(HD, pattern = "Q80", replacement = "HD")
HD <- sub(HD, pattern = "Q92", replacement = "HD")

# transforms mRNA_v to int
m <- mapply(mRNA_v, FUN=as.integer)
rownames(m) <- rownames(mRNA_v)

Samples <- colnames(m)
Conditions <- HD
colData <- cbind(Samples, Conditions)
rownames(colData) <- colnames(m)

dds <- DESeqDataSetFromMatrix(countData = m, colData = colData, 
                              design = ~Conditions)

keep <- rowSums(counts(dds)) >= 50
dds <- dds[keep,]

plotMDS(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))

### normalised counts for ML
dds <- estimateSizeFactors(dds)
sizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)
train_mRNA <- normalized_counts[which(rownames(normalized_counts) %in% rownames(mRNA_consistent) == TRUE),]
valData <- as.data.frame(t(train_mRNA))
valData$Samples <- HD
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\ML_input")
write.table(valData, file= "../../Early Detection/ML_input/validation_mRNA_counts.txt", sep="\t",
            quote=F, col.names=NA)
