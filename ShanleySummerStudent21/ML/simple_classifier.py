from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from ML import *

loc = "C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\InputForML\\"
f1 = "ML_data_train_mRNA.csv"
f2= "ML_data_validate_mRNA.csv"

def classifySVM(train_df, test_df):
    clf = svm.SVC(gamma=0.001, C=100.)
    X_train, Y_train = get_X_y(train_df)
    X_test, Y_test = get_X_y(test_df)
    clf.fit(X_train, Y_train)
    #print(cross_val_score(clf, X, y, cv=5, scoring='recall_macro'))
    Y_preds = clf.predict(X_test)
    print(Y_preds[:15])
    print(Y_test[:15])

    print('Test Accuracy : %.3f'%(Y_preds == Y_test).mean())
    print('Test Accuracy : %.3f'%clf.score(X_test, Y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%clf.score(X_train, Y_train))


train_df = pd.read_csv((loc+f1))
test_df = pd.read_csv((loc+f2))
classifySVM(train_df, test_df)