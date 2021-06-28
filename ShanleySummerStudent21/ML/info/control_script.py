import os
import site
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import zip_longest
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn import metrics
import sklearn
print(sklearn.__version__)

# Add path to sources root to Python's PATH variable
site.addsitedir(os.path.dirname(os.path.dirname(os.path.abspath(''))))
from ML import *

if Import:
    Data = pd.read_csv(Data)
    Data = Data.drop(Data.columns[[0]], axis=1)
    print(Data.shape)

if ImportVal:
    Val = pd.read_csv(ValData)
    Val = Val.drop(Val.columns[[0]], axis=1)
    print(Val.shape)
    array = Val.values
    X_val = array[:, 0:30]
    print(X_val[:1, :4])
    y_val = array[:, 31]
    print(y_val[:25])
    scaler_val = min_max_scaler = preprocessing.MinMaxScaler().fit(X_val)
    X_val_scaled = scaler_val.transform(X_val)


if DataInvestigation:
    print(print(Data.head(20)))
    print(Data.describe())
    print(Data.iloc[:,-1:].head(20))

if split:
    array = Data.values
    X = array[:,0:30]
    print(X[:1,:4])
    y = array[:,31]
    print(y[:25])

if featureSelection:
    selector = SelectKBest(chi2, k=27)
    X_KBest = selector.fit_transform(X, y)
    print(X_KBest.shape)
    X_val_KBest = selector.transform(X_val)
    print(X_val_KBest.shape)

if printSelected:
    mask = selector.get_support()
    #print(selector.get_support())
    feature_names = list(Data.iloc[:,:-1].columns.values)
    new_features = []
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)

    dataframe = pd.DataFrame(X_KBest, columns=new_features)
    #print(dataframe.head(n=5))
    KFeatures = list(dataframe.iloc[:, :-1].columns.values)
    print(list(set(KFeatures) & set(feature_names)))

if printNotSelected:
    print(list(set(feature_names) - set(KFeatures)))

if scaler:
    plt.plot(X)
    plt.savefig(os.path.join(PLOT_DIR, 'unscaled_X_train.png'))
    plt.close()
    scaler = min_max_scaler = preprocessing.MinMaxScaler().fit(X)
    #print(scaler)
    #print(scaler.mean_)
    #print(scaler.scale_)
    X_scaled = scaler.transform(X)
    print(X_scaled)
    print(X_scaled.shape)
    plt.plot(X_scaled)
    plt.savefig(os.path.join(PLOT_DIR, 'scaled_X_train.png'))
    plt.close()

if Alg_tests:
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=20, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_scaled, y, cv=20, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

if plotAlg:
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.savefig(os.path.join(PLOT_DIR, 'M_mRNA_algs.png'))
    plt.close()

if crossVal:
    model = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_scaled, y)
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=1)
    scores = cross_val_score(model, X_scaled, y, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #predicted = cross_val_predict(model, X_KBest, y, cv=10)
    #print(predicted.shape)
    #met = metrics.accuracy_score(y, predicted)
    #print(met)
    predictedVal = cross_val_predict(model, X_val_scaled, y_val, cv=15)
    print(predictedVal.shape)
    metVal = metrics.accuracy_score(y_val, predictedVal)
    print(metVal)

if confusionMatrix:
    print(metrics.confusion_matrix(y_val, predictedVal))
    plot_confusion_matrix(model, X_val_scaled, y_val)
    plt.savefig(os.path.join(PLOT_DIR, 'ConfuMat.png'))
    print(metrics.classification_report(y_val, predictedVal))

if RobustML:
    GeneNames = []
    Accuracies = []
    Predictions = []
    for i in range(1, 68):
        selector = SelectKBest(chi2, k=i)
        X_KBest = selector.fit_transform(X, y)
        X_val_KBest = selector.transform(X_val)

        scaler = preprocessing.StandardScaler().fit(X_KBest)
        X_scaled = scaler.transform(X_KBest)
        X_val_scaled = scaler.transform(X_val_KBest)
        mask = selector.get_support()
        feature_names = list(Data.iloc[:, :-1].columns.values)
        new_features = []
        for bool, feature in zip(mask, feature_names):
            if bool:
                new_features.append(feature)

        dataframe = pd.DataFrame(X_scaled, columns=new_features)
        KFeatures = list(dataframe.iloc[:, :-1].columns.values)
        #print(list(set(KFeatures) & set(feature_names)))
        Genes = list(set(KFeatures) & set(feature_names))
        GeneNames.append(Genes)

        model = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_scaled, y)
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
        scores = cross_val_score(model, X_scaled, y, cv=cv)
        Acc = ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        Accuracies.append(Acc)

        predictedVal = cross_val_predict(model, X_val_KBest, y_val, cv=3)
        metVal = metrics.accuracy_score(y_val, predictedVal)
        Predictions. append(metVal)
        print(Predictions)

