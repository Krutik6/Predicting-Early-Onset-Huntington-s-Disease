"""
There are clear differences between the number of HD, and WT mice. This can be ammended using SMOTE, which artifically increases the number of samples by generating new, but similar samples. For this, I have chosen to use SVM-SMOTE which appears to be the most applicable for this situation, but this SMOTE should be adjusted later on in training, to see how this affects training results.

"""

from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from pandas import read_csv

x = read_csv("fill")
y = read_csv("fill me")
X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))