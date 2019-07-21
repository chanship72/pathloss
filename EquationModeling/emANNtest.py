import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from EquationModeling.equationmodel_ann import ann_mlp_regression, prediction_rmse_error
from EquationModeling.util import ADD_data_loader, combineDF, filteringDF, getFreeSpacePathLoss

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

iksan_fileList = {'../data/PLdata_iksan_wt_400.csv':[('frequency', np.log10(400)), ('heightTB',30), ('heightB',15), ('heightM',2)],
            '../data/PLdata_iksan_wt_1399.csv':[('frequency', np.log10(1399)), ('heightTB',30), ('heightB',15), ('heightM',2)],
            '../data/PLdata_iksan_wt_2249.csv':[('frequency', np.log10(2249)), ('heightTB',30), ('heightB',15), ('heightM',2)]}
nonsan_fileList = {'../data/PLdata_nonsan_wt_400.csv':[('frequency', np.log10(400)), ('heightTB',30), ('heightB',15), ('heightM',2)],
            '../data/PLdata_nonsan_wt_1399.csv':[('frequency', np.log10(1399)), ('heightTB',30), ('heightB',15), ('heightM',2)],
            '../data/PLdata_nonsan_wt_2249.csv':[('frequency', np.log10(2249)), ('heightTB',30), ('heightB',15), ('heightM',2)]}
paju_fileList = {'../data/PLdata_paju_wt_400.csv':[('frequency', np.log10(400)), ('heightTB',100), ('heightB',7), ('heightM',2)],
            '../data/PLdata_paju_wt_1399.csv':[('frequency', np.log10(1399)), ('heightTB',100), ('heightB',7), ('heightM',2)],
            '../data/PLdata_paju_wt_2249.csv':[('frequency', np.log10(2249)), ('heightTB',100), ('heightB',7), ('heightM',2)]}

addIksan = ADD_data_loader(iksan_fileList)
addNonsan = ADD_data_loader(nonsan_fileList)
addPaju = ADD_data_loader(paju_fileList)

# print(addIksan.describe())
# print(addNonsan.describe())
# print(addPaju.describe())

addData = combineDF([addIksan, addNonsan, addPaju])
print("data count:",addData.count())
addData = filteringDF(addData, 'heightTM', [10,100])
print("data count:",addData.count())


addData['freePathloss'] = getFreeSpacePathLoss(addData['distance'],addData['frequency'])
addData = addData[addData['pathloss'] >= addData['freePathloss']]
print("data count:",addData.count())

print("ADD data sample:",addData.head())

print("ADD data description")
print(addData.describe())
print("Covariance Matrix - ADD data")
print(addData.cov())
print("--------------------------------------------------")

# addData.to_csv("/Users/peter.park/workspace/proj_pathloss/EquationModeling/data/adddata.csv")

X = np.array(addData[['logDistance', 'logFrequency', 'logHeightB', 'logHeightM', 'logHeightTratio']])
Y = np.array(addData[['pathloss']])

kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)


modelList = []
i = 1
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model = ann_mlp_regression(X_train, y_train.flatten(), (60,), activation='logistic', optimizer='adam')

    trainError = prediction_rmse_error(model.predict(X_train), y_train)
    testError = prediction_rmse_error(model.predict(X_test), y_test)

    print("ANN-Model-"+str(i)+"-train error(RMSE):", trainError)
    print("ANN-Model-"+str(i)+"-test error(RMSE):", testError)

    modelList.append(model)
    filename = 'model/ann_model_' + str(i) + '.sav'
    pickle.dump(modelList[-1], open(filename, 'wb'))
    i+=1

