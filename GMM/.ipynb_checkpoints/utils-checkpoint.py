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

def gp_train_multi_3dgraph(model, X, Y, Xscatter):
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(6)
    F = np.linspace(np.log10(3400), np.log10(6400), 10)
    conF = []
#     np.zeros((len(Xscatter[:,0]), len(F)))
    for f in F:
        conF.append(np.array([f] * len(Xscatter[:,0])))
    print(np.array(conF).shape)
#     assert F.shape == Xscatter[:,1].shape, "different size"
    
#     conX = np.concatenate((np.array(Xscatter[:,0]).reshape(-1,1),np.array(F).reshape(-1,1)),axis=1)
#     print(conX.shape)
    
#     X_0 = np.linspace(1, 3, num=len(Xscatter))
#     X_0_scatter = X_0.T.reshape(-1,1)
#     X_1 = np.linspace(2, 3, num=len(Xscatter))
#     X_1_scatter = X_1.T.reshape(-1,1)
#     X_all = np.concatenate((X_0_scatter, np.array(Xscatter[:,1]).reshape(-1,1)), axis=1)
    ax = plt.axes(projection='3d')

    group = ['3.4Ghz', '5.3Ghz', '6.4Ghz']
    for idx in range(len(X)):
        ax.plot3D(X[idx][:,0], X[idx][:,1], model.predict(X[idx]),'gray')
        ax.scatter(X[idx][:,0], X[idx][:,1], Y[idx], s=1, label=group[idx], zorder=-1, alpha=0.3);
    y_pred, sigma = model.predict(Xscatter, return_std=True)
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred - 1.9600 * sigma), cmap='Blues', linewidth=0.2, antialiased=True)
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred + 1.9600 * sigma), cmap='Reds', linewidth=0.2, antialiased=True)
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred), cmap='binary', linewidth=0.2, antialiased=True)
    #     ax.plot_surface(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred - 1.9600 * sigma).reshape(-1,1), cmap='binary', linewidth=0.2, antialiased=True)
#     ax.plot_surface(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred + 1.9600 * sigma).reshape(-1,1), cmap='binary', linewidth=0.2, antialiased=True)
#     ax.plot_surface(np.array(Xscatter[:,0]), np.array(F), model.predict(np.array(conF).T), cmap='binary')

#     if flag == 'bh':
#         ax.set_xlim(1.7, 2.8)
#     else:
#         ax.set_xlim(1.7, 3.1)
#     ax.set_ylim(np.log10(3200),np.log10(7000))

    ax.set_xlabel("Log distance(m)",labelpad=18,fontsize=18)
    ax.set_ylabel("Frequency(Ghz)",labelpad=18,fontsize=18)
    ax.set_zlabel("Path Loss(dB)",labelpad=10,fontsize=18)
#     ax.legend(frameon=0, markerscale=5, loc='upper right')
    ax.view_init(elev=20, azim=220)
    
#     ax.xaxis.set_major_locator(mtick.LogLocator(base=10**(1/10)))
#     plt.setp(ax.get_xminorticklabels(), visible=False);

#     plt.xticks([2.0,3.0],[sci_notation(2),sci_notation(3)])
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     labels[1] = sci_notation(2)
#     if flag == 'bh':
#         labels[6] = sci_notation(3)    
#     else:
#         labels[6] = sci_notation(3)
#     ax.set_xticklabels(labels)
    plt.minorticks_on()
    plt.rcParams['xtick.labelsize']=15
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

#     ax.set_xticks([1.6,2.0,2.9],[sci_notation(1),sci_notation(2),sci_notation(3)])
#     ax.set_yticks([np.log10(3400),np.log10(5300),np.log10(6400)],['3.4','5.3','6.4'])

    #plt.yticks([np.log10(3400),np.log10(5300),np.log10(6400)],['3.4','5.3','6.4'],fontsize=18)
    plt.show()
    
def gp_ann_train_multi_3dgraph(modelGp, modelAnn, X, Y, Xscatter):
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(6)

#     X_0 = np.linspace(1, 3, num=len(Xscatter))
#     X_0_scatter = X_0.T.reshape(-1,1)
#     X_1 = np.linspace(2, 3, num=len(Xscatter))
#     X_1_scatter = X_1.T.reshape(-1,1)
#     X_all = np.concatenate((X_0_scatter, np.array(Xscatter[:,1]).reshape(-1,1)), axis=1)
    ax = plt.axes(projection='3d')

    _, sigma = modelGp.predict(Xscatter, return_std=True)
    ann_y_pred = modelAnn.predict(Xscatter)

    group = ['3.4Ghz', '5.3Ghz', '6.4Ghz']
    for idx in range(len(X)):
        _, sigmaT = modelGp.predict(X[idx], return_std=True)
        ann_y_predT = modelAnn.predict(X[idx])

        ax.plot3D(X[idx][:,0], X[idx][:,1], (ann_y_predT - 1.9600 * sigmaT),'gray')
        ax.plot3D(X[idx][:,0], X[idx][:,1], (ann_y_predT - 1.9600 * sigmaT),'gray')
        ax.plot3D(X[idx][:,0], X[idx][:,1], ann_y_predT,'gray')
        ax.scatter(X[idx][:,0], X[idx][:,1], Y[idx], s=1, label=group[idx], zorder=-1, alpha=0.3);
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (ann_y_pred - 1.9600 * sigma), cmap='Blues', linewidth=0.2, antialiased=True)
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (ann_y_pred + 1.9600 * sigma), cmap='Reds', linewidth=0.2, antialiased=True)
    ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (ann_y_pred), cmap='binary', linewidth=0.2, antialiased=True)
    #     ax.plot_surface(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred - 1.9600 * sigma).reshape(-1,1), cmap='binary', linewidth=0.2, antialiased=True)
#     ax.plot_surface(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), (y_pred + 1.9600 * sigma).reshape(-1,1), cmap='binary', linewidth=0.2, antialiased=True)
#     ax.plot_surface(np.array(Xscatter[:,0]), np.array(F), model.predict(np.array(conF).T), cmap='binary')

#     if flag == 'bh':
#         ax.set_xlim(1.7, 2.8)
#     else:
#         ax.set_xlim(1.7, 3.1)
#     ax.set_ylim(np.log10(3200),np.log10(7000))

    ax.set_xlabel("Log distance(m)",labelpad=18,fontsize=18)
    ax.set_ylabel("Frequency(Ghz)",labelpad=18,fontsize=18)
    ax.set_zlabel("Path Loss(dB)",labelpad=10,fontsize=18)
#     ax.legend(frameon=0, markerscale=5, loc='upper right')
    ax.view_init(elev=20, azim=220)
    
#     ax.xaxis.set_major_locator(mtick.LogLocator(base=10**(1/10)))
#     plt.setp(ax.get_xminorticklabels(), visible=False);

#     plt.xticks([2.0,3.0],[sci_notation(2),sci_notation(3)])
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     labels[1] = sci_notation(2)
#     if flag == 'bh':
#         labels[6] = sci_notation(3)    
#     else:
#         labels[6] = sci_notation(3)
#     ax.set_xticklabels(labels)
    plt.minorticks_on()
    plt.rcParams['xtick.labelsize']=15
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

#     ax.set_xticks([1.6,2.0,2.9],[sci_notation(1),sci_notation(2),sci_notation(3)])
#     ax.set_yticks([np.log10(3400),np.log10(5300),np.log10(6400)],['3.4','5.3','6.4'])

    #plt.yticks([np.log10(3400),np.log10(5300),np.log10(6400)],['3.4','5.3','6.4'],fontsize=18)
    plt.show()
# X_train, y_train, X_val, y_val = data_loader_pathloss("../data/PLdata_bh_34.mat")
# X_train, y_train, X_val, y_val
