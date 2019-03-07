from sklearn.model_selection import train_test_split

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

    X = X.reshape((X.shape[0], 1))

    X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X,Y,test_size=0.3, shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test





