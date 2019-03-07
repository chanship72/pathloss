import matplotlib.pyplot as plt
import numpy as np
import json
from random import random

from MLP.utils import data_loader_pathloss

from sklearn.neural_network import MLPRegressor

def mlp_regression(X, Y, hiddel_layer, activation, loss, alpha = 0.01, learning_init=0.01):

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

    mlp = MLPRegressor(hidden_layer_sizes=hiddel_layer,
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

def mlp_prediction_error(model, X, Y):
    X_predictions = model.predict(X)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))

    return rmse

def validation_test(mode, max_layers, h_unit, activation, loss):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_loader_pathloss("PLdata.mat")

    ### Hidden Layer Test
    #   Constant = # of Hidden Layer x # of Hidden Unit = 1000

    rmseList = []
    modelList = []
    layerList = []
    unitList = []

    if(mode == 'hl'):
        for h_layer in range(1, max_layers):
            hidden_layer = (h_unit,) * h_layer
            model = mlp_regression(Xtrain, Ytrain, hidden_layer, activation, loss)
            rmse = mlp_prediction_error(model, Xval, Yval)

            layerList.append(h_layer)
            rmseList.append(rmse)
            modelList.append(model)

    elif mode == 'hu':
        for unit in range(1,h_unit,30):
            hidden_layer = (h_unit,) * max_layers
            model = mlp_regression(Xtrain, Ytrain, hidden_layer, activation, loss)
            rmse = mlp_prediction_error(model, Xval, Yval)

            unitList.append(unit)
            rmseList.append(rmse)
            modelList.append(model)

    min_loss = min(rmseList)

    best_idx = rmseList.index(min_loss)
    best_model = modelList[best_idx]

    testRMSE = mlp_prediction_error(best_model, Xtest, Ytest)

    plt.figure(int(random()*1000))
    best_model.fit(Xtrain, Ytrain)
    plt.scatter(Xtrain, Ytrain)
    plt.plot(np.array(Xtrain), np.array(mlp_prediction(best_model, Xtrain)), color="red")
    plt.title('Trained Model based(' + activation + ' ' + loss + ')')
    plt.xlabel('log distance (m)')
    plt.ylabel('pathloss (dB)')
    plt.tight_layout()
    plt.savefig("dumpDir/regression_" + mode + "_" + activation + "_" + loss + ".png")
    plt.figure(figsize=(12, 5))
    plt.show()

    plt.figure(int(random()*1000))
    if mode == 'hl':
        plt.plot(np.array(layerList), np.array(rmseList), color="green")
        plt.xlabel('# of hidden layers')
    elif mode == 'hu':
        plt.plot(np.array(unitList), np.array(rmseList), color="green")
        plt.xlabel('# of hidden units')

    plt.ylabel('Root Mean Square Error(RMSE)')

    plt.title('RMSE trend <' + activation + ',' + loss + '> | ' + "RMSE(test) : " + str(testRMSE))
    plt.figure(figsize=(12, 5))
    plt.savefig("dumpDir/trend_" + mode + "_" + activation + "_" + loss + ".png")

    # save file
    if mode == 'hl':
        json.dump({'train loss': min_loss, 'val_key': layerList, 'val_val': rmseList, 'test': testRMSE},
              open('dumpDir/MLP_lr' + "_" + mode + "_" + activation + "_" + loss + '.json', 'w'))

    elif mode == 'hu':
        json.dump({'train loss': min_loss, 'val_key': unitList, 'val_val': rmseList, 'test': testRMSE},
              open('dumpDir/MLP_lr' + "_" + mode + "_" + activation + "_" + loss + '.json', 'w'))
    plt.show()
