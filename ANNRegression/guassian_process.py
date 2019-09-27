import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

pd.set_option('display.max_rows', 999)
pd.set_option('precision', 5)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# from MLP.mlp_regression import mlp_regression, model_validation, mlp_prediction, mlp_prediction_error, mlp_train_graph, errorDist, mlp_train_multi_graph, mlp_train_multi_graph_comb, mlp_train_multi_3dgraph_comb
# from MLP.utils import combineArray, multiArraySort, data_loader_from_csv, data_loader_pathloss, describeData, data_loader_pathloss_with_freq

pd.set_option('display.max_rows', 999)
pd.set_option('precision', 5)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.neural_network import MLPRegressor

def gp_regression(kernel='RBF', length = 1.0):

    # Instantiate a Gaussian Process model
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    if kernel == 'RBF':
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3)) \
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+2))
#         kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2)) \
#                 + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-2, 1e+2))
    elif kernel == 'RQ':
        kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.5)\
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    elif kernel == 'ESS':
        kernel = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0))\
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)
    
    return gp_model

def gp_train_graph(model, X, Y, xCategory = ('0.4Ghz', '1.399Ghz', '2.249Ghz')):
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

def gp_linear_compare_graph(GPmodel, LinearModel, X, Y, xCategory = ('0.4Ghz GP', '0.4Ghz Linear', '1.399Ghz GP', '1.399Ghz Linear', '2.249Ghz GP', '2.249Ghz Linear')):
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

        GPPred = GPmodel.predict(elementX)
        LinearPred = LinearModel.predict(elementX)
    
        #plt.scatter(originX, elementY, s=1)
        plt.plot(elementX[:,0], GPPred, color=cmap(cmap_i))
        plt.plot(elementX[:,0], LinearPred, color=cmap(cmap_i), linestyle='dashed')
        cmap_i += 0.8

    plt.xlabel("log distance(Mhz)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(xCategory)
    plt.show()
    
def prediction_rmse_error(pred, Y):
    rmse = np.sqrt(np.mean(np.power(Y - pred, 2)))

    return rmse

