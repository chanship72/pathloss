import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick
import matplotlib.tri as mtri

from math import floor, log10
from matplotlib.ticker import FormatStrFormatter
from utils import data_loader_pathloss
from matplotlib.lines import Line2D
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def mlp_regression(X, Y, hidden_layer, activation, loss, alpha = 0.0, learning_init=0.001):

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
                                           max_iter=2000,
                                           learning_rate_init=learning_init,
                                           alpha=alpha,
                                           tol = 1e-6,
                                           verbose=False)

    mlp.fit(X,Y)
    return mlp

def mlp_prediction(model, X):
    return model.predict(X)

def build_tf_ann_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.sigmoid, input_shape=[2]),
        layers.Dense(64, activation=tf.nn.sigmoid),
        layers.Dense(1)
      ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mape',
                optimizer=optimizer,
                metrics=['mse', 'mae', 'mape'])
    print(model.summary())
    return model

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

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

def mlp_multiple_models_train_graph(models, X, Y):
    cmap = plt.cm.coolwarm
    plt.figure(figsize=(16, 6))
    plt.scatter(X[:,0], Y, s=1)
    cmap_i = 0.0
    
    for model in models:
        plt.plot(X[:,0], model.predict(X), color=cmap(cmap_i))
        cmap_i += 0.8
    plt.xlabel("Distance(km) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(('Linear', 'ANN-MLP', 'Gaussian-Process'))
    plt.show()
    
def ann_train_graph(X, Y, pred, Xscatter, Yscatter, var=False, std=0):
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(6)

    plt.scatter(Xscatter, Yscatter, s=1)    
    print(std)
    if var:
        plt.fill_between(X[:,0], pred - std, pred + std, alpha=0.1, color='k')
        plt.plot(X[:,0], pred, color='b')
    else:
        plt.plot(X[:,0], pred, color='b')

    plt.xlabel("Distance(km) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(('ANN-Learning', 'data', 'confidence range-99.7%'))
    plt.show()

        
def mlp_train_multi_graph(X, Y, pred, Xscatter, Yscatter, type, var=False, std=[]):
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(6)
    cmap_i = 0.0
    plt.scatter(Xscatter, Yscatter, s=1)    
    for idx in range(len(X)):
#         plt.scatter(X[idx][:,0], Y[idx], s=1, c=cmap(cmap_i))
        if var:
            plt.fill_between(X[idx][:,0], pred[idx] - std[idx], pred[idx] + std[idx], alpha=0.1, color='r')
            plt.plot(X[idx][:,0], pred[idx], color=cmap(cmap_i))
        else:
            plt.plot(X[idx][:,0], pred[idx], color=cmap(cmap_i))
        cmap_i += 0.8

    if type == 'distance':
        plt.xlabel("Distance(km) - log(x)")
    elif type == 'height':
        plt.xlabel("Distance(km)")
    else:
        plt.xlabel("Distance(km) - log(x)")
    
    plt.ylabel("Path Loss(dB)")
    plt.legend(('0.4Ghz', '1.399Ghz', '2.249Ghz'))
    plt.show()

def mlp_train_multi_graph_comb(model, X, Xscatter, Yscatter, activation, loss):
    cmap = plt.cm.coolwarm
    fig,ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    cmap_i = 0.0
#     ax.set_xscale('log')
    plt.scatter(Xscatter, Yscatter, s=1)        
    for idx in range(len(X)):
        plt.plot(X[idx][:,0], model.predict(X[idx]), color=cmap(cmap_i))        
        cmap_i += 0.8
#     ax.set_xlabel("Distance(m) [$10^{x}]$",fontsize=12)
    plt.xlabel("Distance(m) - log(x)")
    plt.xscale('log')
    plt.ylabel("Path Loss(dB)")
    plt.legend(('3.4Ghz', '5.3Ghz', '6.4Ghz'))
    plt.show()
    
def sci_notation(num):
    return "$10^{}$".format(num)

def myticks(x,pos):
    if x == 0: return "$0$"

    exponent = int(x)
    coeff = 10**x/10**exponent

    return r"${:2.2f} \times 10^{{ {:2d} }}$".format(coeff,exponent)

def mlp_train_multi_3dgraph_comb(model, X, Y, Xscatter, freqRange = ['3.4','5.3','6.4'], sigma=False, colormap='binary'):
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)
    min_dist = min(Xscatter[:,0])
    max_dist = max(Xscatter[:,0])
    min_freq = min(Xscatter[:,1])
    max_freq = max(Xscatter[:,1])
    dist = np.linspace(min_dist,max_dist, num=100)
    freq = np.linspace(min_freq,max_freq, num=100)

    dist, freq = np.meshgrid(dist, freq)
    x, y = dist.flatten(), freq.flatten()
    X_all = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)
    if sigma:
        _, z = model.predict(X_all, return_std=True)
    else:
        z = model.predict(X_all)
#     print(z)
#     print("dist:" + str(dist.shape))
#     print("freq:" + str(freq.shape))
#     print("X_all:" + str(X_all.shape))
#     X_0 = np.linspace(1, 3, num=len(Xscatter))
#     X_0_scatter = X_0.T.reshape(-1,1)
#     X_1 = np.linspace(2, 3, num=len(Xscatter))
#     X_1_scatter = X_1.T.reshape(-1,1)
#     X_all = np.concatenate((X_0_scatter, np.array(Xscatter[:,1]).reshape(-1,1)), axis=1)
    ax = plt.axes(projection='3d')
#     tri = mtri.Triangulation(dist, freq)
#     group = ['3.4Ghz', '5.3Ghz', '6.4Ghz']
    group = [freq+'Ghz' for freq in freqRange]
    if sigma == False:
        for idx in range(len(X)):
            ax.plot3D(X[idx][:,0], X[idx][:,1], model.predict(X[idx]),'gray')
            ax.scatter(X[idx][:,0], X[idx][:,1], Y[idx], s=1, label=group[idx], zorder=-1, alpha=0.3);
#     ax.plot_trisurf(np.array(Xscatter[:,0]), np.array(Xscatter[:,1]), model.predict(Xscatter),cmap='binary', alpha=0.5)
    ax.plot_trisurf(x, y, z ,cmap=colormap, alpha=0.5)
    
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
    if sigma == False:
        ax.set_zlim(-50, 200)
#     ax.set_xticks([1.6,2.0,2.9],[sci_notation(1),sci_notation(2),sci_notation(3)])
#     ax.set_yticks([np.log10(3400),np.log10(5300),np.log10(6400)],['3.4','5.3','6.4'])
        
#     plt.yticks([np.log10(float(freqRange[0])*1000),np.log10(float(freqRange[1])*1000),np.log10(float(freqRange[2])*1000)],freqRange,fontsize=18)
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
#             print("#hidden_layer:" + str(hidden_layer) + "hidden_units:" + str(unit) + "RMSE:" + str(rmse))
            print(str(unit) + "      " + str(rmse))

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
