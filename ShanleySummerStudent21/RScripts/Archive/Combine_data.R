library(factoextra)
setwd("C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\ML_input")
miRNA <- read.csv("../../Early Detection/ML_input/miRNA_data.csv", row.names = 1)
colnames(miRNA) <- gsub(colnames(miRNA), pattern = "\\.", replacement = "-")
mRNA <- read.csv("../../Early Detection/ML_input/mRNA_data.csv", row.names = 1)

#mRNA <- mRNA[which(rownames(mRNA) %in% rownames(miRNA)),]
#miRNA <- miRNA[which(rownames(miRNA) %in% rownames(mRNA)),]
mRNA$Samples <- NULL

intersect(rownames(mRNA), rownames(miRNA))
Data <- cbind(mRNA, miRNA)
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
# input for ML - training data
Data$Samples <- Samples
Data$Samples <- sub(Data$Samples, pattern = "_10m", replacement = "")
Data$Samples <- sub(Data$Samples, pattern = "_6m", replacement = "")
write.csv(Data, "../../Early Detection/ML_input/DE_ML_Data.csv", row.names = TRUE)

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
write.csv(Data, "../../Early Detection/ML_input/DE_ML_Data_val.csv", row.names = TRUE)
print("finished")