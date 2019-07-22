import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

# from MLP.mlp_regression import mlp_regression, model_validation, mlp_prediction, mlp_prediction_error, mlp_train_graph, errorDist, mlp_train_multi_graph, mlp_train_multi_graph_comb, mlp_train_multi_3dgraph_comb
# from MLP.utils import combineArray, multiArraySort, data_loader_from_csv, data_loader_pathloss, describeData, data_loader_pathloss_with_freq

pd.set_option('display.max_rows', 999)
pd.set_option('precision', 5)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.neural_network import MLPRegressor

def ann_mlp_regression(X, Y, hidden_layer, activation, optimizer, alpha = 0.0001, learning_init=0.01):

    """
    mlp = MLPRegressor(hidden_layer_sizes=(1000,),
                                           activation='tanh',
                                           solver='lbfgs',
                                           learning_rate='constant',
                                           max_iter=1000,
                                           learning_rate_init=0.01,
                                           alpha=0.01,
                                           verbose=True)
    """

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer,
                                           activation=activation,
                                           solver=optimizer,
                                           learning_rate='constant',
                                           max_iter=1000,
                                           learning_rate_init=learning_init,
                                           alpha=alpha,
                                           tol = 1e-4,
                                           verbose=False)
    mlp.fit(X,Y)
    print("loss:", mlp.loss_)
    return mlp

def ann_train_graph(model, X, Y, xCategory = ('0.4Ghz', '1.399Ghz', '2.249Ghz')):
    #   X: array of distance list
    #   Y: array of pathloss list
    #   xCategory: title of each X set
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(6)

    # print("X:",X)
    # print("Y:",Y)
    cmap_i = 0.0
    for idx in range(len(X)):
#         print(X[idx].head())
        minX = min(np.array(X[idx])[:,0])
        maxX = max(np.array(X[idx])[:,0])
#         print("min:", minX)
#         print("max:", maxX)

        originX = np.array(X[idx])[:,0]

#         print(len(np.array(X[idx])))

        linX = np.linspace(minX, maxX, num=len(np.array(X[idx])))
        X[idx]['logDistance'] = linX
    # linY = np.linspace(min(Y),max(Y), num=500)
        elementX = np.array(X[idx])
        elementY = np.array(Y[idx])

#         print(elementX.shape)
#         print(elementY.shape)

        pred = model.predict(elementX)

        #plt.scatter(originX, elementY, s=1)
        plt.plot(elementX[:,0], pred, color=cmap(cmap_i))
        cmap_i += 0.8

    plt.xlabel("log distance(Mhz)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(xCategory)
    plt.show()

def ann_linear_compare_graph(ANNmodel, LinearModel, X, Y, xCategory = ('0.4Ghz ANN', '0.4Ghz Linear', '1.399Ghz ANN', '1.399Ghz Linear', '2.249Ghz ANN', '2.249Ghz Linear')):
    #   X: array of distance list
    #   Y: array of pathloss list
    #   xCategory: title of each X set
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(6)

    # print("X:",X)
    # print("Y:",Y)
    cmap_i = 0.0
    for idx in range(len(X)):
#         print(X[idx].head())
        minX = min(np.array(X[idx])[:,0])
        maxX = max(np.array(X[idx])[:,0])
#         print("min:", minX)
#         print("max:", maxX)

        originX = np.array(X[idx])[:,0]

#         print(len(np.array(X[idx])))

        linX = np.linspace(minX, maxX, num=len(np.array(X[idx])))
        X[idx]['logDistance'] = linX
    # linY = np.linspace(min(Y),max(Y), num=500)
        elementX = np.array(X[idx])
        elementY = np.array(Y[idx])

#         print(elementX.shape)
#         print(elementY.shape)

        ANNPred = ANNmodel.predict(elementX)
        LinearPred = LinearModel.predict(elementX)
    
        #plt.scatter(originX, elementY, s=1)
        plt.plot(elementX[:,0], ANNPred, color=cmap(cmap_i))
        plt.plot(elementX[:,0], LinearPred, color=cmap(cmap_i), linestyle='dashed')
        cmap_i += 0.8

    plt.xlabel("log distance(Mhz)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(xCategory)
    plt.show()
    
def prediction_rmse_error(pred, Y):
    rmse = np.sqrt(np.mean(np.power(Y - pred, 2)))

    return rmse

# print("<Iksan - Winter> Relu (dist<=3000m)")
# mlp_train_multi_3dgraph_comb(model, dataX_m, dataY_m, X_train_m, ['0.4', '1.399','2.249'])

# print(model.loss_)
# print(model.n_iter_ )
# print(model.n_layers_)
# print(model.n_outputs_)
# y_pred_400 = model.predict(X_val_m_400)
# y_pred_1399 = model.predict(X_val_m_1399)
# y_pred_2249 = model.predict(X_val_m_2249)
# y_pred = model.predict(X_val_m)
# stat = np.array([mlp_prediction_error(model,X_val_m_400, y_val_m_400),mean_absolute_error(y_pred_400, y_val_m_400),mean_absolute_percentage_error(y_pred_400, y_val_m_400),mean_squared_log_error(y_pred_400, y_val_m_400),r2_score(y_pred_400, y_val_m_400),
#                  mlp_prediction_error(model,X_val_m_1399, y_val_m_1399),mean_absolute_error(y_pred_1399, y_val_m_1399),mean_absolute_percentage_error(y_pred_1399, y_val_m_1399),mean_squared_log_error(y_pred_1399, y_val_m_1399),r2_score(y_pred_1399, y_val_m_1399),
#                  mlp_prediction_error(model,X_val_m_2249, y_val_m_2249),mean_absolute_error(y_pred_2249, y_val_m_2249),mean_absolute_percentage_error(y_pred_2249, y_val_m_2249),mean_squared_log_error(y_pred_2249, y_val_m_2249),r2_score(y_pred_2249, y_val_m_2249),
#                  mlp_prediction_error(model,X_val_m, y_val_m), mean_absolute_error(y_pred, y_val_m),mean_absolute_percentage_error(y_pred, y_val_m),mean_squared_log_error(y_pred, y_val_m),r2_score(y_pred, y_val_m)])
#
# (pd.DataFrame(stat.reshape((4,5)),index=pd.Index(['0.4Ghz','1.399Ghz','2.249Ghz', 'Overall']), columns=pd.Index(['RMSE','MAE','MAPE','MSLE','R2'],name='TEST ERROR(dB)')))
#
# model = pickle.load(open('model/ann_model_1.sav', 'rb'))
#
# ann_train_graph(model, X, Y, 'log distance(KM)', [np.log10(400), np.log10(1399), np.log10(2249)], xColumn = ('0.4Ghz', '1.399Ghz', '2.249Ghz'))