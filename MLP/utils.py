import numpy as np
import pandas as pd
import json
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy import stats
### Loss functions ###

# Softmax loss and Softmax gradient
class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output

# L2 mean square loss and L2 gradient
class mean_square_loss:
    def __init__(self):
        self.expand_Y = None

    def forward(self, X, Y):
        forward_output = np.sqrt(np.mean(np.power(Y-X,2)))
        return forward_output

    def backward(self, grad, Y):
        if np.sqrt(np.mean(np.power(Y-grad,2))) > 0:
            backward_output = -np.sqrt(np.mean(Y-grad))/np.sqrt(np.mean(np.power(Y-grad,2)))
        else:
            backward_output = -np.sqrt(np.mean(Y - grad))

        backward_output = np.ones(grad.shape)*backward_output
        return backward_output

### Momentum ###
def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum


def data_loader_mnist(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = np.array(train_set[0])
    Ytrain = np.array(train_set[1])
    Xvalid = np.array(valid_set[0])
    Yvalid = np.array(valid_set[1])
    Xtest = np.array(test_set[0])
    Ytest = np.array(test_set[1])

    return Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest

def data_loader_pathloss(dataset):
    mat_contents = np.array(sio.loadmat(dataset)['temp1'])
    # print(mat_contents.shape)

    d = mat_contents[:,0]
    p = mat_contents[:,1]
    # print(d,p)

#     X = np.log10(d)
    X = d
    Y = p

    # X = X.reshape((X.shape[0], 1))

    X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.2, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=0.5, shuffle=True)

    df_train = pd.DataFrame({'X_train':X_train, 'y_train':y_train}).sort_values(by=['X_train'])
    df_val = pd.DataFrame({'X_val':X_val, 'y_val':y_val}).sort_values(by=['X_val'])
    df_test = pd.DataFrame({'X_test':X_test, 'y_test':y_test}).sort_values(by=['X_test'])

    X_train = np.array(df_train['X_train'])
    y_train = np.array(df_train['y_train'])

    X_val = np.array(df_val['X_val'])
    y_val = np.array(df_val['y_val'])

    X_test = np.array(df_test['X_test'])
    y_test = np.array(df_test['y_test'])

    return X_train.reshape(-1, 1), y_train, X_val.reshape(-1, 1), y_val, X_test.reshape(-1, 1), y_test

def data_loader_pathloss_with_freq(dataset, freq):
    mat_contents = np.array(sio.loadmat(dataset)['temp1'])
    # print(mat_contents.shape)

    d = mat_contents[:,0].reshape(-1, 1)
    p = mat_contents[:,1]
    f = np.array([freq]*len(d)).reshape(-1, 1)

    X = np.concatenate((np.log10(d),f),axis=1)
    Y = p

    X_train, X_val, y_train, y_val = train_test_split(X,Y,test_size=0.2, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=0.5, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

def predict_label(f):
    # This is a function to determine the predicted label given scores
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))

def predict_value(model, X):
    ### Computing validation accuracy ###

    a1 = model['L1'].forward(X)
    h1 = model['nonlinear1'].forward(a1)
    d1 = model['drop1'].forward(h1, is_train=False)
    y_hat = model['L2'].forward(d1)

    return y_hat

def describeData(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, label='Test'):
    pd.options.display.max_rows = 999
    dic = {
           '1.X_train':pd.Series(Xtrain.flatten()), 
           '2.y_train':pd.Series(Ytrain.flatten()), 
           '3.X_val':pd.Series(Xval.flatten()),
           '4.y_val':pd.Series(Yval.flatten()),
           '5.X_test':pd.Series(Xtest.flatten()),
           '6.y_test':pd.Series(Ytest.flatten()),
          }
    df_bh_34 = pd.DataFrame(dic)
    print(label)
    print("------------------------------------------------------------------------")
    print(df_bh_34.describe())
    print()

class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape, 1

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY
    

    
