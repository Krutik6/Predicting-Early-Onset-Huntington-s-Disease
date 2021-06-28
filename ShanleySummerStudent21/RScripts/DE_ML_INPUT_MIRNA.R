# load libs
library(DESeq2)
library(limma)
library(dplyr)
library(tidyverse) 
library(factoextra)
#set wd
setwd("~/Documents/HD/Data/Early_detection/Data")
#load data
miRNA <- read.csv("miRNA_train.csv", row.names = 1)
colnames(miRNA)
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
HD

# If a third of the samples (37) had a rowSum 
# because many miRNAs are low level and we have many samples
X <- miRNA[!rowSums(miRNA < 50) >= 37, , drop = FALSE]

m <- mapply(X, FUN=as.integer)
rownames(m) <- rownames(X)

Samples <- colnames(m)
Conditions <- HD
colData <- cbind(Samples, Conditions)
rownames(colData) <- colnames(m)

dds <- DESeqDataSetFromMatrix(countData = m, colData = colData, 
                              design = ~Conditions)

#keep <- rowSums(counts(dds)) >= 1000
#dds <- dds[keep,]

plotMDS(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))
boxplot(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))

### normalised counts for ML
dds <- estimateSizeFactors(dds)
sizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)
write.table(normalized_counts, file="normalized_miRNA_counts.txt", sep="\t",
            quote=F, col.names=NA)

###
dds$Conditions <- factor(dds$Conditions, levels = unique(dds$Conditions))
dds$Conditions
dds <- DESeq(dds)
resultsNames(dds)

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
hist(M6$padj)
M10 <- getDE(numC = 'HD_10m', denC = 'WT_10m')
hist(M10$padj)
################################################################################
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
norm_miRNA <- read.table("normalized_miRNA_counts.txt", row.names = 1)
colnames(norm_miRNA) <- colnames(miRNA)
miRNA_consistent <- norm_miRNA[which(rownames(norm_miRNA) %in% Sig_genes$Names),] 

#PCA plots
Data <- t(miRNA_consistent)
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
setwd("~/Documents/HD/Data/Early_detection/ML_input")
write.csv(DF, "miRNA_data.csv")
################################################################################
#validation data
setwd("~/Documents/HD/Data/Early_detection/Data")
miRNA_v <- read.csv("miRNA_validation.csv", row.names = 1)
pheno_v <- read.csv("pheno_validation.csv", row.names = 1)

HD <- sub(pheno_v$Name, pattern = "fe", replacement = "")
HD <- sub(HD, pattern = "male_", replacement = "")
HD <- sub(HD, pattern = "Q20", replacement = "WT")
HD <- sub(HD, pattern = "Q111", replacement = "HD")
HD <- sub(HD, pattern = "Q140", replacement = "HD")
HD <- sub(HD, pattern = "Q175", replacement = "HD")
HD <- sub(HD, pattern = "Q80", replacement = "HD")
HD <- sub(HD, pattern = "Q92", replacement = "HD")

m <- mapply(miRNA_v, FUN=as.integer)
rownames(m) <- rownames(miRNA_v)

Samples <- colnames(m)
Conditions <- HD
colData <- cbind(Samples, Conditions)
rownames(colData) <- colnames(m)

dds <- DESeqDataSetFromMatrix(countData = m, colData = colData, 
                              design = ~Conditions)

plotMDS(dds@assays@data@listData$counts, col = as.numeric(dds$Conditions))

### normalised counts for ML
dds <- estimateSizeFactors(dds)
sizeFactors(dds)
normalized_counts <- counts(dds, normalized=TRUE)
train_miRNA <- normalized_counts[which(rownames(normalized_counts) %in% rownames(miRNA_consistent) == TRUE),]
valData <- as.data.frame(t(train_miRNA))
valData$Samples <- HD
setwd("~/Documents/HD/Data/Early_detection/ML_input")
write.table(valData, file="validation_miRNA_counts.txt", sep="\t",
            quote=F, col.names=NA)

