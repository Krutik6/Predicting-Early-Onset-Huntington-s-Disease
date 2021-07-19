from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from ML import *
from sklearn.preprocessing import StandardScaler
import glob
"""
This script aids visualisation at each stage in the code pipeline
"""

def visualise(x,df, fname):
    loc = r"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Figures\\Data_Exploration\\"

    x = x.drop(columns = "Samples")

    # PCA can't be performed on one feature
    if len(x.columns) == 1:
        pair_plots(df, loc, fname)
        return False

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # get components
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                               , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['Conditions']]], axis = 1)


    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ["HD", "WT"]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Conditions'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()

    fig.savefig((loc+"PCA_{}.png".format(fname)))
    print("saved", "PCA_{}.png".format(fname), "in", loc)
    pair_plots(df, loc, fname)

def pair_plots(df, loc, fname):
    sns.set_style("whitegrid");
    df = df.drop(columns="Samples")
    df = df.drop(columns="Unnamed: 0")
    fig = sns.pairplot(df, hue="Conditions")
    print("reached")
    fig.savefig((loc+"pair_plot_{}.png".format(fname)))
    print("saved", "pair_plot_{}.png".format(fname), "in", loc)

def box_plot(df,loc ,fname):
    df = prepare_for_box_plot(df)
    sns.set_theme(style="ticks", palette="pastel")
    y=sns.scatterplot(x="variable", y="value", hue = "Conditions",palette=["m", "g"], data=df)
    #sns.despine(offset=10, trim=True)
    #y.legend([],[], frameon=False)
    y.figure.savefig((loc+"scatter_{}.png".format(fname)))



def prepare_for_box_plot(df):
    con = df["Conditions"]
    df = df.drop(columns=["Unnamed: 0", "Samples", "Conditions"])
    m = df.melt(ignore_index=False)
    merged = m.merge(con, left_index=True, right_index=True)
    return merged

dir = r"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\FilteredData\\"
os.chdir(dir)

loc = r"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Figures\\Data_Exploration\\"

for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        df = pd.read_csv(f)
        x,y = get_X_y(df)
        name = filename.replace(".csv", "")
        name = name+"_filtered"
        box_plot(df, loc, name)
        #visualise(x, df, name)

dir = r"C:\\Users\\Colle\\OneDrive\\Documents\\Boring\\2021 Summer Internship\\ShanleySummerStudent21\\Early Detection\\Data\\Preprocessed_Data\\"
os.chdir(dir)
"""
for filename in glob.glob('*.csv'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        df = pd.read_csv(f)
        x,y = get_X_y(df)
        name = filename.replace(".csv", "")
        name = name+"_all_data"
        visualise(x, df, name)
"""
