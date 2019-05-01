import numpy as np
import pandas as pd
import json
import scipy.io as sio
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange','blue','red','green','black','gray'])


def data_loader_pathloss(dataset):
    mat_contents = np.array(sio.loadmat(dataset)['temp1'])
    # print(mat_contents.shape)

    d = np.array(mat_contents[:, 0]).reshape(-1, 1)
    p = np.array(mat_contents[:, 1]).reshape(-1, 1)
    d = np.log10(d)
    X = np.concatenate((d, p), axis=1)

    return X

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)

def data_loader_pathloss(dataset):
    mat_contents = np.array(sio.loadmat(dataset)['temp1'])
    # print(mat_contents.shape)

    d = mat_contents[:, 0]
    p = mat_contents[:, 1]

    X = np.log10(d)
    Y = p

    scaler = StandardScaler()
    print(scaler.fit(Y.reshape(-1, 1)))

    print(scaler.scale_)
    print(scaler.n_samples_seen_)

    print("----------------------------------")
    tY = np.multiply(scaler.transform(Y.reshape(-1, 1)).reshape(len(Y), ), 10)
    tYd = pd.DataFrame(tY)
    # tY = 10*tY
    tY = tY.astype(np.int64)
    print(tY.astype(np.int64))
    print(tYd.describe())
    # matplotlib histogram
    plt.hist(tY, color='blue', edgecolor='black',
             bins=int(180 / 5))

    # # seaborn histogram
    sns.distplot(tY, hist=True, kde=False,
                 bins=int(180 / 5), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Histogram of Category')
    plt.xlabel('Category (pathloss)')
    plt.show()

    print(tY)

    Y = tY

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

    df_train = pd.DataFrame({'X_train': X_train, 'y_train': y_train}).sort_values(by=['X_train'])
    df_val = pd.DataFrame({'X_val': X_val, 'y_val': y_val}).sort_values(by=['X_val'])
    return np.array(df_train['X_train']).reshape(-1, 1), np.array(df_train['y_train']), np.array(
        df_val['X_val']).reshape(-1, 1), np.array(df_val['y_val'])



# X_train, y_train, X_val, y_val = data_loader_pathloss("../data/PLdata_bh_34.mat")
# X_train, y_train, X_val, y_val
