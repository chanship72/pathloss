import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from util import inverseScale
# from MLP.mlp_regression import mlp_regression, model_validation, mlp_prediction, mlp_prediction_error, mlp_train_graph, errorDist, mlp_train_multi_graph, mlp_train_multi_graph_comb, mlp_train_multi_3dgraph_comb
# from MLP.utils import combineArray, multiArraySort, data_loader_from_csv, data_loader_pathloss, describeData, data_loader_pathloss_with_freq

pd.set_option('display.max_rows', 999)
pd.set_option('precision', 5)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def ann_mlp_regression(hidden_layer, activation, optimizer, alpha = 0.01, learning_init=0.001):

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
                                           max_iter=2000,
                                           learning_rate_init=learning_init,
                                           alpha=alpha,
                                           tol = 1e-6,
                                           verbose=False)
    return mlp

def errorDist(yhat, y):
    error = yhat - y
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [dB]")
    _ = plt.ylabel("Count")
    plt.show()
    df_error = pd.DataFrame({'Error(Noise) Distribution': error})
    print(df_error.describe())

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
        # minMaxScale for log distance(log_d)
        minXlogD = X[idx]['logDistance'].min()
        minXlogHB = X[idx].loc[X[idx]['logDistance'] == minXlogD]['logHeightB']
        maxXlogD = X[idx]['logDistance'].max()
        maxXlogHB = X[idx].loc[X[idx]['logDistance'] == maxXlogD]['logHeightB']

        linXlogD = np.linspace(minXlogD, maxXlogD, num=len(np.array(X[idx])))
        # minMaxScale for log_hb * log_d
        linXlogAD = np.multiply(linXlogD, np.array(X[idx]['logHeightB']))
        # set input as random points of (1 + log_hb)log_d
        X[idx]['logDistance'] = linXlogD
        X[idx]['logAntennaMulLogDistance'] = linXlogAD
        elementX = np.array(X[idx])
        elementY = np.array(Y[idx])

        ANNPred = ANNmodel.predict(elementX)
        LinearPred = LinearModel.predict(elementX)
        
        #plt.scatter(originX, elementY, s=1)
        plt.plot(elementX[:,0], ANNPred, color=cmap(cmap_i))
        plt.plot(elementX[:,0], LinearPred, color=cmap(cmap_i), linestyle='dashed')
        cmap_i += 0.8

    plt.xlabel("log distance(KM)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(xCategory)
    plt.show()
    
def prediction_rmse_error(pred, Y):
    rmse = np.sqrt(np.mean(np.power(Y - pred, 2)))

    return rmse

def mlp_train_graph(model, X, Y, activation, loss):
    plt.figure(figsize=(15, 6))
    plt.scatter(X[:,0].reshape(-1,1), Y, s=1)
    X.sort(axis=0)
    plt.plot(X[:,0].reshape(-1,1), model.predict(X), color="red")
    plt.xlabel("Distance(m) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.show()
    
def polynomialRegression():
    i = 1
    for X_train,y_train,X_test,y_test, trainError, testError in dataSet:
        ndeg = 2
        polynomial_model = None
        polynomial_features = None
        X_train_poly = None
        polynomial_features = PolynomialFeatures(degree=ndeg)
        X_train_poly = polynomial_features.fit_transform(X_train)

        ANNmodel = pickle.load(open("model/ann_model_"+ str(i) +".sav", 'rb'))
        ANNpred = ANNmodel.predict(X_train)

        polynomial_model =  LinearRegression(fit_intercept=False).fit(X_train_poly, ANNpred)

        PolyPred = polynomial_model.predict(X_train_poly)

        similarity = np.sqrt(mean_squared_error(PolyPred, ANNpred))
        print("Similarity(ANN-Poly):", similarity)
        # (a,b) => a, b, a^2, ab, b^2
        # (a, b, c, d, e, f) => 1, a, b, c
        #                       d, e, f, a^2
        #                       b^2, c^2, d^2, e^2
        #                       f^2, ab, ac, ad,
        #                       be, bf, cd, ce,
        #                       cf, de, df, ef,
        print("L_pathloss = {:6.2f}log_d + {:6.2f}log_f + {:6.2f}log_hb + {:6.2f}log_hm+ {:6.2f}log_(h_tb/h_tm) + {:6.2f}log_d*log_hb +{:6.2f}(log_d)^2 + {:6.2f}(log_f)^2 + {:6.2f}(log_hb)^2 + {:6.2f}log_hm^2 + {:6.2f}(log_(h_tb/h_tm))^2 + {:6.2f}(log_d*log_hb)^2 + {:6.2f}log_d*log_f + {:6.2f}log_d*log_hb + {:6.2f}log_d*log_hm + {:6.2f}log_d*log_(h_tb/h_tm)+ {:6.2f}log_d*log_d*log_hb + {:6.2f}log_f*log_hb + {:6.2f}log_f*log_hm + {:6.2f}log_f*log_(h_tb/h_tm)+ {:6.2f}log_f*log_d*log_hb + {:6.2f}log_hb*log_hm + {:6.2f}log_hb*log_(h_tb/h_tm) + {:6.2f}log_hb*log_d*log_hb + {:6.2f}log_hm*log_(h_tb/h_tm) + {:6.2f}log_hm*log_d*log_hb + {:6.2f}log_(h_tb/h_tm)*log_d*log_hb + {:6.2f}".format(polynomial_model.coef_[0],polynomial_model.coef_[1],polynomial_model.coef_[2],polynomial_model.coef_[3],polynomial_model.coef_[4],polynomial_model.coef_[5],polynomial_model.coef_[6],polynomial_model.coef_[7],polynomial_model.coef_[8],polynomial_model.coef_[9],polynomial_model.coef_[10],polynomial_model.coef_[11],polynomial_model.coef_[12],polynomial_model.coef_[13],polynomial_model.coef_[14],polynomial_model.coef_[15],polynomial_model.coef_[16],polynomial_model.coef_[17],polynomial_model.coef_[18],polynomial_model.coef_[19],polynomial_model.coef_[20],polynomial_model.coef_[21],polynomial_model.coef_[22],polynomial_model.coef_[23],polynomial_model.coef_[24],polynomial_model.coef_[25],polynomial_model.coef_[26],polynomial_model.coef_[27],polynomial_model.intercept_))
        print()
        i+=1