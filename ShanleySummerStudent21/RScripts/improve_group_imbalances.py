"""
There are clear differences between the number of HD, and WT mice. This can be ammended using SMOTE, which artifically increases the number of samples by generating new, but similar samples. For this, I have chosen to use SVM-SMOTE which appears to be the most applicable for this situation, but this SMOTE should be adjusted later on in training, to see how this affects training results.

"""

from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from pandas import read_csv
from varname import nameof

from ML import *



def perform_smote(RNA, X_file_name, y_file_name):
    X, y = get_X_y(RNA)
    # todo change to svm smote
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
    # save files
    loc = "../InputForML/SMOTE/{}.csv"

    X_resampled.to_csv(loc.format(X_file_name))
    y_resampled.to_csv(loc.format(y_file_name))
    print("saved", X_file_name, y_file_name)

train_mRNA = read_csv("../InputForML/ML_data_train_mRNA.csv")
train_miRNA = read_csv("../InputForML/ML_data_train_miRNA.csv")
validate_mRNA = read_csv("../InputForML/ML_data_validate_mRNA.csv")
validate_miRNA = read_csv("../InputForML/ML_data_validate_miRNA.csv")

RNAs = [train_mRNA, train_miRNA, validate_mRNA, validate_miRNA]
RNA_labs = ["train_mRNA", "train_miRNA", "validate_mRNA", "validate_miRNA"]

i=0
for rna in RNAs:
    perform_smote(rna, ("X_"+RNA_labs[i]), ("y_"+RNA_labs[i]))
    i+=1
