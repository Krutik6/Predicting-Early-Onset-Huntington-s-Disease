# performs feature extraction on the datasets using variance threshold method

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

"""
I have standardised the variances by diving by the means, in order to reduce skew


From towardsdatascience.com -> https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
"It is not fair to compare the variance of a feature to another. The reason is that as the values in the distribution get bigger, the variance grows exponentially. In other words, the variances will not be on the same scale. 

One method we can use is normalizing all features by dividing them by their mean. This method ensures that all variances are on the same scale. "

"""


def selectFeatures(RNA_train_data, train_file_name, RNA_validatation_data, validate_file_name):
    # transposes the data so that the RNAs are columns
    labelled_RNAs_train = RNA_train_data.transpose()
    labelled_RNAs_validate = RNA_validatation_data.transpose()

    # sets mRNAs as columns
    labelled_RNAs_train, labelled_RNAs_train.columns = labelled_RNAs_train[1:], labelled_RNAs_train.iloc[0]
    labelled_RNAs_validate, labelled_RNAs_validate.columns = labelled_RNAs_validate[1:], labelled_RNAs_validate.iloc[0]

    RNA_counts_train = labelled_RNAs_train
    RNA_counts_validate = labelled_RNAs_validate

    print(RNA_counts_train.shape)

    # removes any RNAs where all counts are 0
    # zero_map =  (RNA_counts_train != 0).all()
    #RNA_counts_validate = RNA_counts_validate[zero_map[1]]
    #RNA_counts_train1 = RNA_counts_train[zero_map[1]]

    RNA_counts_train_zeroed = RNA_counts_train.loc[:, (RNA_counts_train != 0).any(axis=0)]
    RNA_counts_validate_zeroed = RNA_counts_validate.loc[:, (RNA_counts_train != 0).any(axis=0)]

#print(RNA_counts_train == RNA_counts_train1)
    # match the removed columns here
    print(RNA_counts_train.shape)

    normalized_df_train = RNA_counts_train_zeroed / RNA_counts_train_zeroed.mean()

    """
    change this for increased / decreased sensitivity
    also consider recusive ft selection
    """
    vt = VarianceThreshold(threshold=1)

    # Fit
    _ = vt.fit(normalized_df_train)

    # Get the mask
    mask = vt.get_support()

    # Subset the DataFrame
    RNA_final_train = RNA_counts_train_zeroed.loc[:, mask]
    RNA_final_validate = RNA_counts_validate_zeroed.loc[:, mask]

    print(RNA_final_train.shape)

    """consider using alternative, compound feature selection including recursive ft selection to further fine tune 
    the model 
    
    """

    RNA_train_t = RNA_final_train.transpose()
    RNA_validate_t = RNA_final_validate.transpose()

    RNA_train_t.to_csv("../Early Detection/Data/" + train_file_name)
    RNA_validate_t.to_csv("../Early Detection/Data/" + validate_file_name)

    print("saved files", train_file_name, validate_file_name)


mRNA_train_data = pd.read_csv(r"../Early Detection/Data/mRNA_train.csv")
mRNA_validate_data = pd.read_csv(r"../Early Detection/Data/mRNA_validation.csv")

miRNA_data = pd.read_csv(r"../Early Detection/Data/miRNA_train.csv")
miRNA_validate_data = pd.read_csv(r"../Early Detection/Data/miRNA_validation.csv")

selectFeatures(mRNA_train_data, "sig_mRNA_train.csv", mRNA_validate_data, "sig_mRNA_validate.csv" )
selectFeatures(miRNA_data, "sig_miRNA_train.csv", miRNA_validate_data, "sig_miRNA_validate.csv")

