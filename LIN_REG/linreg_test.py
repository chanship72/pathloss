from linreg_regularization import linear_regression_noreg, linear_regression_invertible, regularized_linear_regression, tune_lambda, root_mean_square_error,mapping_data
from data_loader import data_loader_pathloss
import numpy as np
import pandas as pd

filename = 'PLdata.mat'


Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, False, False, 0)
w = linear_regression_noreg(Xtrain, ytrain)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mse = root_mean_square_error(w, Xtrain, ytrain)
print("RMSE on train is %.5f" % mse)
mse = root_mean_square_error(w, Xval, yval)
print("RMSE on val is %.5f" % mse)
mse = root_mean_square_error(w, Xtest, ytest)
print("RMSE on test is %.5f" % mse)


Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, False, False, 0)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  " + str(bestlambd))
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mse = root_mean_square_error(w, Xtrain, ytrain)
print("RMSE on train is %.5f" % mse)
mse = root_mean_square_error(w, Xval, yval)
print("RMSE on val is %.5f" % mse)
mse = root_mean_square_error(w, Xtest, ytest)
print("RMSE on test is %.5f" % mse)

Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, True, False, 0)
w = linear_regression_invertible(Xtrain, ytrain)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mse = root_mean_square_error(w, Xtrain, ytrain)
print("RMSE on train is %.5f" % mse)
mse = root_mean_square_error(w, Xval, yval)
print("RMSE on val is %.5f" % mse)
mse = root_mean_square_error(w, Xtest, ytest)
print("RMSE on test is %.5f" % mse)

Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, True, False, 0)
w = regularized_linear_regression(Xtrain, ytrain, 0.1)
print("dimensionality of the model parameter is ", w.shape, ".", sep="")
print("model parameter is ", np.array_str(w))
mse = root_mean_square_error(w, Xtrain, ytrain)
print("RMSE on train is %.5f" % mse)
mse = root_mean_square_error(w, Xval, yval)
print("RMSE on val is %.5f" % mse)
mse = root_mean_square_error(w, Xtest, ytest)
print("RMSE on test is %.5f" % mse)

print("if your maaping function is correct, simplely change the 'power' value to see how MSE change when 'power' changes")
power = 2
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, False, True, power)
bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
print("Best Lambda =  ", bestlambd, sep="")
w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
print("dimensionality of the model parameter is ", len(w), ".", sep="")
print("model parameter is ", np.array_str(w))
mse = root_mean_square_error(w, Xtrain, ytrain)
print("RMSE on train is %.5f" % mse)
mse = root_mean_square_error(w, Xval, yval)
print("RMSE on val is %.5f" % mse)
mse = root_mean_square_error(w, Xtest, ytest)
print("RMSE on test is %.5f" % mse)

print("if your maaping function is correct, simplely change the 'power' value to see how MSE change when 'power' changes")
power = 20
for i in range(2, power):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = data_loader_pathloss(filename, False, True, i)
    bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval)
    print('best lambd is ' + str(bestlambd))
    w = regularized_linear_regression(Xtrain, ytrain, bestlambd)
    mse = root_mean_square_error(w, Xtrain, ytrain)
    print('when power = ' + str(i))
    print("RMSE on train is %.5f" % mse)
    mse = root_mean_square_error(w, Xval, yval)
    print("RMSE on val is %.5f" % mse)
    mse = root_mean_square_error(w, Xtest, ytest)
    print("RMSE on test is %.5f" % mse)




