import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from os import chdir
import sklearn
from sklearn import metrics, datasets, neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)

# Global variables are always in caps, to distinguish them from local variables.
WORKING_DIRECTORY = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIRECTORY, 'Data')
INFO_DIR = os.path.join(WORKING_DIRECTORY, 'Info')
PLOT_DIR = os.path.join(INFO_DIR, "PLOT")
Data = os.path.join(DATA_DIR, "ML_Data.csv")
ValData = os.path.join(DATA_DIR, "ML_Data_val.csv")

# Flags that change the behaviour of the control_script

# flag to demonstrate the principle of flags
Import = True
ImportVal = True
DataInvestigation = False
split = True
featureSelection = False
printSelected = False
printNotSelected = False
scaler = True
Alg_tests = True
plotAlg = True
crossVal = True
confusionMatrix = True
RobustML = False

def get_X_y(df):
    X = df.loc[:, df.columns != 'Conditions']
    X = X.loc[:, X.columns != "Unnamed: 0"]
    y = df.loc[:, df.columns == 'Conditions']
    return X, y

def open_files(f, filename):
    X_train = pd.read_csv(f)
    X_train = X_train.drop(columns="Unnamed: 0")

    y_train_v = filename.replace("X", "y")
    y_train = pd.read_csv(y_train_v)
    y_train = y_train.drop(columns="Unnamed: 0")

    X_test_v = filename.replace("train", "validate")
    X_test = pd.read_csv(X_test_v)
    X_test = X_test.drop(columns="Unnamed: 0")

    y_test_v = filename.replace("X_train", "y_validate")
    y_test = pd.read_csv(y_test_v)
    y_test = y_test.drop(columns="Unnamed: 0")
    return X_train, X_test, y_train, y_test

def permutation_based_feature_importance(clf, X, y, feature_names, save=False, filename = False):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=0,
                                    n_jobs=-1)

    fig, ax = plt.subplots()
    sorted_idx = result.importances_mean.argsort()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feature_names[sorted_idx])
    ax.set_title("Permutation Importance of each feature")
    ax.set_ylabel("Features")
    fig.tight_layout()
    if not save:
        plt.show()
    else:
        if not filename:
            raise FileNotFoundError ("Provide a filename to save the file")
        else:
            loc = r"../../Early Detection/Data/Figures/"
            plt.savefig((loc+filename))
def scoreModel():
    pass