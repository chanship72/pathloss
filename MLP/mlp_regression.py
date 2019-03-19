import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from random import random

from MLP.utils import data_loader_pathloss
from matplotlib.lines import Line2D
from sklearn.neural_network import MLPRegressor

def mlp_regression(X, Y, hidden_layer, activation, loss, alpha = 0.01, learning_init=0.01):

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
                                           solver=loss,
                                           learning_rate='constant',
                                           max_iter=1000,
                                           learning_rate_init=learning_init,
                                           alpha=alpha,
                                           verbose=False)

    mlp.fit(X,Y)
    return mlp

def mlp_prediction(model, X):
    return model.predict(X)

def errorDist(yhat, y):
    error = yhat - y
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [dB]")
    _ = plt.ylabel("Count")
    plt.show()
    df_error = pd.DataFrame({'Error(Noise) Distribution': error})
    print(df_error.describe())

def mlp_prediction_error(model, X, Y):
    X_predictions = model.predict(X)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))
    
    return rmse

def mlp_train_graph(model, X, Y, activation, loss, rmse):
    plt.figure(figsize=(12, 5))
    plt.scatter(X, Y)
    plt.plot(np.array(X), np.array(mlp_prediction(model, X)), color="red")
    plt.xlabel("log distance (m)")
    plt.ylabel("Path Loss(dB)")
    plt.title("Trained Model based(" + activation + ", " + loss + ") RMSE:" + str(rmse))
    plt.show()

def mlp_train_multi_graph(X, Y, pred, Xscatter, Yscatter, activation, loss):
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    cmap_i = 0.0
#     ax.set_xscale('log')
    plt.scatter(Xscatter, Yscatter, s=1)        
    for idx in range(len(X)):
        plt.plot(X[idx], pred[idx], color=cmap(cmap_i))        
        cmap_i += 0.8
#     ax.set_xlabel("Distance(m) [$10^{x}]$",fontsize=12)
    plt.xlabel("Distance(m) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(('3.4Ghz', '5.3Ghz', '6.4Ghz'))
    plt.show()
    
def model_validation(Xtrain, Ytrain, Xval, Yval, mode, max_layers, max_unit, activation, loss):
    # Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_loader_pathloss("PLdata.mat")

    ### Hidden Layer Test
    #   Constant = # of Hidden Layer x # of Hidden Unit = 1000

    rmseList = []
    modelList = []
    layerList = []
    unitList = []

    if(mode == 'hl'):
        for h_layer in range(1, max_layers):
            hidden_layer = (max_unit,) * h_layer
            model = mlp_regression(Xtrain, Ytrain, hidden_layer, activation, loss)
            rmse = mlp_prediction_error(model, Xtrain, Ytrain)

            layerList.append(h_layer)
            rmseList.append(rmse)
            modelList.append(model)
            print("#hidden_layer: " + str(hidden_layer) + " / hidden_units:" + str(max_unit) + " / RMSE:" + str(rmse))

    elif mode == 'hu':
        for unit in range(1,max_unit):
            hidden_layer = (unit) * max_layers
            model = mlp_regression(Xtrain, Ytrain, hidden_layer, activation, loss)
            rmse = mlp_prediction_error(model, Xtrain, Ytrain)

            unitList.append(unit)
            rmseList.append(rmse)
            modelList.append(model)
            # print("#hidden_layer:" + str(hidden_layer) + "hidden_units:" + str(unit) + "RMSE:" + str(rmse))

    min_loss = min(rmseList)

    best_idx = rmseList.index(min_loss)
    best_model = modelList[best_idx]
    plt.figure(figsize=(12, 5))
    # testRMSE = mlp_prediction_error(best_model, Xtest, Ytest)
    plt.title('RMSE trend <' + activation + ',' + loss + '> | ' + "best RMSE : " + str(min_loss))
    if mode == 'hl':
        plt.plot(np.array(layerList), np.array(rmseList), color="green")
        plt.xlabel('# of hidden layers')
    elif mode == 'hu':
        plt.plot(np.array(unitList), np.array(rmseList), color="green")
        plt.xlabel('# of hidden units')
    plt.ylabel('Root Mean Square Error(RMSE)')

    plt.show()

    return best_model
    # plt.savefig("dumpDir/trend_" + mode + "_" + activation + "_" + loss + ".png")

    # save file
    # if mode == 'hl':
    #     json.dump({'train loss': min_loss, 'val_key': layerList, 'val_val': rmseList, 'test': testRMSE},
    #           open('dumpDir/MLP_lr' + "_" + mode + "_" + activation + "_" + loss + '.json', 'w'))
    #
    # elif mode == 'hu':
    #     json.dump({'train loss': min_loss, 'val_key': unitList, 'val_val': rmseList, 'test': testRMSE},
    #           open('dumpDir/MLP_lr' + "_" + mode + "_" + activation + "_" + loss + '.json', 'w'))
