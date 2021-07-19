from sklearn.model_selection import train_test_split
from glob import glob
import os
import pandas as pd

# select age data and split into test, and training files
loc = "../Early Detection/Data/Preprocessed_Data/"#

for filename in glob('*_[0-9]+.csv'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        df = pd.read_csv(f)
        name = filename.replace(".csv", "")
        train_test_split(df, test_size=0.2)
