import numpy as np
import pandas as pd
import json
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
import seaborn as sns


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

def data_loader_from_csv(dataset, k, strategy):
    df = pd.read_csv(dataset, delimiter=',', names=["type", "dist", "ploss", "height"])
    df['dist'] = df['dist'] * 1000
    X = np.array(df['dist'].apply(np.log10))
    Y = np.array(df['ploss'])

    # scaling X
    # scaler = StandardScaler()
    # scaler.fit(X.reshape(-1,1))
    # X = scaler.transform(X.reshape(-1,1))
    # X = np.array(X).flatten()

    enc = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy=strategy)
    tY = enc.fit_transform(Y.reshape(-1, 1))
    rangeX = enc.bin_edges_[0]
    for i in range(k):
        print("range" + str(i+1) + ":" + str(rangeX[i].astype(int)) + "~" + str(rangeX[i + 1].astype(int)))

    # --------------------------------------------------
    # histogram
    plt.figure(figsize=(10, 5))
    #     plt.hist(tY, color='blue', edgecolor='black', bins=int(100/k))

    # # seaborn histogram
    sns.distplot(tY, hist=True, kde=False,
                 bins=int(k), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Histogram of Category')
    plt.xlabel('Range (pathloss)')
    plt.xticks(())
    plt.show()
    # --------------------------------------------------

    Y = tY.flatten()
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

    df_train = pd.DataFrame({'X_train': X_train, 'y_train': y_train}).sort_values(by=['X_train'])
    df_val = pd.DataFrame({'X_val': X_val, 'y_val': y_val}).sort_values(by=['X_val'])
    return np.array(df_train['X_train']).reshape(-1, 1), np.array(df_train['y_train']), np.array(
        df_val['X_val']).reshape(-1, 1), np.array(df_val['y_val'])

## X_train, y_train, X_val, y_val

def build_dataframe(model, y_hat, y_val, cY, X_val, k):
    y_hat = model.predict(X_val)
    cY = model.predict_proba(X_val) * 100
    colLabel = ['range' + str(i + 1) for i in range(k)]
    cY_df = pd.DataFrame(cY.astype(int), columns=colLabel)
    cY_df[:] = cY_df[:].astype(str) + '%'
    cY_df['X(log distance)'] = X_val
    cY_df['prediction'] = y_hat.astype(int) + 1
    cY_df['real value'] = y_val.astype(int) + 1

    return cY_df