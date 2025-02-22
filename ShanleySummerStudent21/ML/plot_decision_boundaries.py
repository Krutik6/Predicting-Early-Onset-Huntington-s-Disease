from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from ML import *

class DecisionBoundaries:
    """
    plots the decision boundaries on data when transformed to 2d using pca (it is impossible to visualise data in more than three dimensions, and often there are more than 3 parameters).

    this will not work when all data is from one category as seen in some mRNA data, it returns an error message in this case but continues to process the other data

    The model is default linear svm, this can be changed when calling, alongside model name for naming

    if calling plot_all, specify loc when calling
    """
    def __init__(self):
        self.loc = ""
        self.model = svm.SVC(kernel='linear')
        self.model_name = "linear SVC"

    def plot(self, X, y, fname):
        le = LabelEncoder()
        y = le.fit_transform(y)
        pca = PCA(n_components=2)
        Xreduced = pca.fit_transform(X)
        try:
            clf = self.model.fit(Xreduced, y)
            fig, ax = plt.subplots()
            # title for the plots
            # Set-up grid for plotting.
            X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
            xx, yy = self._make_meshgrid(X0, X1)

            self._plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('Decision surface using PCA transformed/projected features with', self.model_name)
            ax.legend()
            plt.savefig(fname)
        except ValueError:
            print("ValueError: The number of classes has to be greater than one; got 1 class")


    def plot_all(self):
        chdir(self.loc)
        for filename in glob.glob('*X*'):
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                X_train, X_test, y_train, y_test = get_age_files(f, filename)
                X = X_train
                y = y_train
                le = LabelEncoder()
                y = le.fit_transform(y)
                pca = PCA(n_components=2)
                Xreduced = pca.fit_transform(X)
                try:
                    clf = self.model.fit(Xreduced, y)
                    fig, ax = plt.subplots()
                    # title for the plots
                    title = ('Decision surface of', self.model_name)
                    # Set-up grid for plotting.
                    X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
                    xx, yy = self._make_meshgrid(X0, X1)

                    self._plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                    ax.set_ylabel('PC2')
                    ax.set_xlabel('PC1')
                    ax.set_xticks(())
                    ax.set_yticks(())
                    ax.set_title('Decision surface using the PCA transformed/projected features')
                    ax.legend()
                    plt.show()
                except ValueError:
                    print("Error displaying",filename)
                    print("ValueError: The number of classes has to be greater than one; got 1 class")


    def _make_meshgrid(self,x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def _plot_contours(self,ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out



