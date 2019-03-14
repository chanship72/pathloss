from sklearn.model_selection import train_test_split
from scipy import stats

import json
import numpy as np
import pandas as pd
import scipy.io as sio

def data_loader_pathloss(dataset):
    mat_contents = np.array(sio.loadmat(dataset)['temp1'])
    # print(mat_contents.shape)

    d = mat_contents[:,0]
    p = mat_contents[:,1]
    # print(d,p)

    X = np.log10(d)
    Y = p

    # X = X.reshape((X.shape[0], 1))

    X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.2, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=0.5, shuffle=True)

    df_train = pd.DataFrame({'X_train':X_train, 'y_train':y_train}).sort_values(by=['X_train'])
    df_val = pd.DataFrame({'X_val':X_val, 'y_val':y_val}).sort_values(by=['X_val'])
    df_test = pd.DataFrame({'X_test':X_test, 'y_test':y_test}).sort_values(by=['X_test'])

    X_train = np.array(df_train['X_train'])
    y_train = np.array(df_train['y_train'])
    X_train = X_train.reshape((X_train.shape[0], 1))

    X_val = np.array(df_val['X_val'])
    y_val = np.array(df_val['y_val'])
    X_val = X_val.reshape((X_val.shape[0], 1))

    X_test = np.array(df_test['X_test'])
    y_test = np.array(df_test['y_test'])
    X_test = X_test.reshape((X_test.shape[0], 1))

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = data_loader_pathloss('PLdata.mat')

print(stats.describe(X_train))
print(stats.describe(X_val))
print(stats.describe(X_test))

