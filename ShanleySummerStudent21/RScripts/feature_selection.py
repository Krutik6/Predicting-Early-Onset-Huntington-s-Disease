# performs feature extraction on the datasets using variance threshold method

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


"""
From towardsdatascience.com -> https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
It is not fair to compare the variance of a feature to another. The reason is that as the values in the distribution get bigger, the variance grows exponentially. In other words, the variances will not be on the same scale. 

One method we can use is normalizing all features by dividing them by their mean. This method ensures that all variances are on the same scale:
"""
RNA_data = pd.read_csv(r"../InputForRFiltering/mRNA_counts.csv")

# transposes the data so that the mRNAs are columns
labelled_RNAs = RNA_data.transpose()
# sets mRNAs as columns
labelled_RNAs, labelled_RNAs.columns = labelled_RNAs[1:], labelled_RNAs.iloc[0]

RNA_counts = labelled_RNAs
print(RNA_counts.shape)

RNA_labels = labelled_RNAs.columns

# removes any RNAs where all counts are 0
RNA_counts = RNA_counts.loc[:, (RNA_counts != 0).any(axis=0)]
print(RNA_counts.shape)

normalized_df = RNA_counts / RNA_counts.mean()

"""
change this for increased / decreased sensitivity
also consider recusive ft selection
"""
vt = VarianceThreshold(threshold=0.1)

# Fit
_ = vt.fit(normalized_df)

# Get the mask
mask = vt.get_support()

# Subset the DataFrame
RNA_final = RNA_counts.loc[:, mask]
print(RNA_final.shape)



"""
consider using alternative, compound feature selection including recursive ft selection to further fine tune the model

"""

RNA_t = RNA_final.transpose()

RNA_t.to_csv("../InputForRFiltering/sig_mRNA_counts.csv")