import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics, datasets, neighbors

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

def scoreModel():
    pass