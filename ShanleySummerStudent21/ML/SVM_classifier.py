from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn import svm
from ML import *
from ML import plot_confusion
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import numpy as np
import io


loc = r"../InputForML/SMOTE/"
chdir(loc)

def classifySVM(save_document=True):
    clf = svm.SVC(gamma=0.001, C=100., probability=True)
    for filename in glob.glob('*X*'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            X_train, X_test, y_train, y_test = get_age_files(f, filename, remove_duplicates=False)

            clf.fit(X_train, y_train)
            #print(cross_val_score(clf, X, y, cv=5, scoring='recall_macro'))
            y_preds = clf.predict(X_test)

            if save_document:
                document = Document()
                document.add_heading("Classifier Performance on {} data".format(filename.replace("X_", "").replace("_", " ").replace("train.csv", ""), "classifier"),0)
                document.add_heading("Training", level=2)
                document.add_paragraph(('Training Accuracy : %.3f'%clf.score(X_train, y_train)))


            print('Training Accuracy : %.3f'%clf.score(X_train, y_train))
            print(filename)



            rna = filename.replace("X_", "").replace(".csv", "")
            name = "SVM_{}_Permutation_Importance.png".format(rna)
            permutation_based_feature_importance(clf, X_train, y_train, X_train.columns, save=True, filename = name)
            return clf, X_test, y_test, y_preds, ["HD", "WT"], document






# todo extract to init
def evaluate_model(y_pred, y_true, X_test, y_test, clf, target_names, document=None):
    ######################################################
    # accuracy

    print("Accuracy: ", accuracy_score(y_true, y_pred))

    ###################################################
    # balanced accuracy
    #
    print("Balanced accuracy score: ", balanced_accuracy_score(y_true, y_pred))

    #########################
    # cohen_kappa_score
    """
    The kappa score is a number between -1 and 1. Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels)
    """
    print("cohen kappa score: ",cohen_kappa_score(y_true, y_pred), "above 0.8 is good agreement")

    ##############################
    # plot confusion matrix
    plot_confusion(clf, X_test, y_test, ["HD", "WT"])
    ####################################
    # classification report

    print("classification report: \n", classification_report(y_true, y_pred, target_names=target_names))
    #########################################
    # idk

    print("Precision: ",metrics.precision_score(y_true, y_pred, average="binary", pos_label="HD"))

    print("Recall:", metrics.recall_score(y_true, y_pred, average="binary", pos_label="HD"))

    print("F1:",metrics.f1_score(y_true, y_pred, average="binary", pos_label="HD"))

    print("F beta, beta-0.5", metrics.fbeta_score(y_true, y_pred, beta=0.5,average="binary", pos_label="HD"))

    print("F beta, beta-1",metrics.fbeta_score(y_true, y_pred, beta=1,average="binary", pos_label="HD"))

    print("F beta, beta-2",metrics.fbeta_score(y_true, y_pred, beta=2,average="binary", pos_label="HD"))

    print("precision recall fscore support", metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5,average="binary", pos_label="HD"))


    # ROC curve
    y_scores = clf.predict_proba(X_test)[:, 1]
    precision, recall, threshold = precision_recall_curve(y_true, y_scores, pos_label="HD")


    print("Average precision score: ", average_precision_score(y_true, y_scores, pos_label="HD"))

    #######################################
    # ROC
    # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
    y_score = clf.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    ##################################
    # precision recall curve

    prec, recall, _ = precision_recall_curve(y_test, y_score,
                                             pos_label=clf.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec,     recall=recall).plot()

    # combine plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.show()

    if document is not None:
        document.add_heading("Test Metrics", level=2)
        document.add_paragraph(("Accuracy: {}".format(accuracy_score(y_true, y_pred))), style = "List Bullet")
        document.add_paragraph(("Balanced accuracy score: {}".format(balanced_accuracy_score(y_true, y_pred))), style = "List Bullet")
        document.add_paragraph(("Cohen kappa score: {} ".format(accuracy_score(y_true, y_pred))), style = "List Bullet")
        p=document.add_paragraph("(The kappa score is a number between -1 and 1. Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels))", style = "List Bullet")
        p.add_run('italic.').italic = True


        # confusion matricies
        document.add_heading("Confusion Matrices", level=2)
        memfile = io.BytesIO()
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clf, X_test, y_test,
                                         display_labels=["HD", "WT"],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            plt.savefig(memfile)
            document.add_picture(memfile, width=Inches(5))
        memfile.close()

        # classification report
        document.add_heading("Classification report", level=2)
        document.add_paragraph("{}".format(classification_report(y_true, y_pred, target_names=target_names)))

        # Precision/recall
        document.add_heading("Precision/Recall Scores", level=2)
        document.add_paragraph("Precision: {}".format(metrics.precision_score(y_true, y_pred, average="binary", pos_label="HD")), style= "List Bullet")
        document.add_paragraph("Recall: {}".format(metrics.recall_score(y_true, y_pred, average="binary", pos_label="HD")), style= "List Bullet")
        document.add_paragraph("F1 {}".format(metrics.f1_score(y_true, y_pred, average="binary", pos_label="HD")), style= "List Bullet")
        document.add_paragraph("Precision Recall F-Score Support: {}".format(metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5,average="binary", pos_label="HD")))


        memfile = io.BytesIO()
        y_score = clf.decision_function(X_test)

        # precision recall curve
        prec, recall, _ = precision_recall_curve(y_test, y_score,
                                                 pos_label=clf.classes_[1])
        pr_display = PrecisionRecallDisplay(precision=prec,     recall=recall).plot()

        # combine plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        plt.savefig(memfile)
        document.add_picture(memfile, width=Inches(5))
        memfile.close()

        document.save(r'../../ML/Classifiers/demo.docx')



##################################



clf, X_test, y_test, y_preds, classes , document = classifySVM()
evaluate_model(y_preds, y_test, X_test, y_test, clf, classes, document=document)