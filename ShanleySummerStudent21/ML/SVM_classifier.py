from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from ML import *
from sklearn.metrics import average_precision_score



loc = r"../InputForML/SMOTE/"
chdir(loc)

def classifySVM():
    clf = svm.SVC(gamma=0.001, C=100.)
    for filename in glob.glob('*X_train*'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            X_train, X_test, y_train, y_test = open_files(f, filename)
            clf.fit(X_train, y_train)
            #print(cross_val_score(clf, X, y, cv=5, scoring='recall_macro'))
            y_preds = clf.predict(X_test)
            x =np.array(y_test.Conditions)

            #print('Test Accuracy : %.3f'%(y_preds == y_test).mean())
            #print('Test Accuracy : %.3f'%clf.score(X_test, y_test)) ## Score method also evaluates accuracy for classification models.
            print('Training Accuracy : %.3f'%clf.score(X_train, y_train))
            rna = filename.replace("X_train_", "").replace(".csv", "")
            name = "SVM_{}_Permutation_Importance.png".format(rna)
            permutation_based_feature_importance(clf, X_train, y_train, X_train.columns, save=True, filename = name)



def plot_confusion():
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import plot_confusion_matrix

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()



def evaluate_model(y_pred, y_true, X_test, y_test, clf):
    ######################################################
    # accuracy
    import numpy as np
    from sklearn.metrics import accuracy_score
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    accuracy_score(y_true, y_pred)

    ###################################################
    # balanced accuracy
    #
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy_score(y_true, y_pred)

    #########################
    # cohen_kappa_score
    """
    The kappa score is a number between -1 and 1. Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)
    """
    from sklearn.metrics import cohen_kappa_score
    cohen_kappa_score(y_true, y_pred)

    ##############################
    # plot confusion matrix
    plot_confusion()
    ####################################
    # classification report
    from sklearn.metrics import classification_report
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 0, 2, 1, 0]
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true, y_pred, target_names=target_names))
    #########################################
    # idk
    from sklearn import metrics
    y_pred = [0, 1, 0, 0]
    y_true = [0, 1, 0, 1]
    metrics.precision_score(y_true, y_pred)

    metrics.recall_score(y_true, y_pred)

    metrics.f1_score(y_true, y_pred)

    metrics.fbeta_score(y_true, y_pred, beta=0.5)

    metrics.fbeta_score(y_true, y_pred, beta=1)

    metrics.fbeta_score(y_true, y_pred, beta=2)

    metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)



    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, threshold = precision_recall_curve(y_true, y_scores)


    print(average_precision_score(y_true, y_scores))

#######################################
    # ROC
    # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
    from sklearn.metrics import roc_curve
    from sklearn.metrics import RocCurveDisplay
    y_score = clf.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    ##################################
    # precision recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import PrecisionRecallDisplay

    prec, recall, _ = precision_recall_curve(y_test, y_score,
                                         pos_label=clf.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec,     recall=recall).plot()

    # combine plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.show()


classifySVM()