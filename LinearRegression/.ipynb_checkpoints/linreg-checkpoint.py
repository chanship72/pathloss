import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


from LIN_REG.data_loader import data_loader_pathloss

def linear_regression_multi_graph(X, Y, pred, Xscatter, Yscatter):
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
#     ax.set_xlabel("Distance(m) - log(x)", fontsize=12)
    plt.xlabel("Distance(m) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(('3.4Ghz', '5.3Ghz', '6.4Ghz'))
    plt.show()

def linear_regression_multi_graph_comb(model, X, Xscatter, Yscatter):
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
#     ax.set_xlabel("Distance(m) - log(x)", fontsize=12)
    plt.xlabel("Distance(m) - log(x)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(('3.4Ghz', '5.3Ghz', '6.4Ghz'))
    plt.show()    
    
def lin_prediction_error(model, X, Y):
#     X = np.log10(X)
    X_predictions = model.predict(X)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))
    
    return rmse

def L_hata(d, f, h_t):
    # Hata model prediction
    A = 46.3 + 33.9*np.log10(f) - 13.28*np.log10(h_t)
    B = -3.2*np.log10(np.power(11.75*2,2)) + 4.97
    C = 44.9 - 6.55*np.log10(h_t)
    D = 0

    return A + B + C * np.log10(0.001 * d) + D

def hata_prediction_error(X, Y, h_t):
    X_predictions = L_hata(X[:,0], X[:,1], h_t)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))
    
    return rmse    

def L_modified_hata(d,f,h_t,area = 'A'):
    # modified Hata model prediction
    d_1, d_2 = 0.0,0.0
    if area == 'A':
        d_1 = -3.53 * np.log10(h_t)
        d_2 = -10 * np.log10(f/1000) - 9
    elif area == 'B':
        d_1 = 2 * np.log10(h_t)
        d_2 = -0.009 * np.exp(f/1000) - 2.3 
        
    A = 46.3 + 33.9*np.log10(f) - 13.28*np.log10(h_t)
    B = -3.2*np.log10(np.power(11.75*2,2)) + 4.97
    C = 44.9 - 6.55*np.log10(h_t)
    D = 0

    return A + B + (C+d_1)*np.log10(0.001*d) + D + d_2


def mhata_prediction_error(X, Y, h_t, area='A'):
    X_predictions = L_modified_hata(X[:,0], X[:,1], h_t, area)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))
    
    return rmse    


def linearRegression(X_train, y_train):
#     X_train = np.log10(X_train)
    
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    return model


#     train_prediction = model.predict(X_train)
#     train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));

#     val_prediction = model.predict(X_val)
#     val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));

#     test_prediction = model.predict(X_test)
#     test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));

#     plt.figure(figsize=(12, 5))
#     ax = plt.axes()
#     ax.scatter(X_train, y_train)
#     ax.plot(X_train, train_prediction, color="red")
#     plt.title("<" + label + ">\nRMSE(train) =" + str(train_rmse)
#               + "\nRMSE(val) =" + str(val_rmse)
#               + "\nRMSE(test) =" + str(test_rmse)
#               + "\ncoefficient:" + str(model.coef_)
#               + "\nbias:" + str(model.intercept_),loc='left')
#     ax.set_xlabel('log distance (m)')
#     ax.set_ylabel('pathloss (dB)')

#     plt.show()
    
def ridgeRegression(X_train, y_train):
    model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    model.fit(X_train, y_train)

    return model    
#     train_prediction = model.predict(X_train)
#     train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));

#     val_prediction = model.predict(X_val)
#     val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));

#     test_prediction = model.predict(X_test)
#     test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));

#     plt.figure(figsize=(12, 5))
#     ax = plt.axes()
#     ax.scatter(X_train, y_train)
#     ax.plot(X_train, train_prediction, color="red")
#     plt.title("<" + label + ">\nRMSE(train) =" + str(train_rmse)
#               + "\n RMSE(val) =" + str(val_rmse)
#               + "\n RMSE(test) =" + str(test_rmse)
#               + "\n coefficient:" + str(model.coef_)
#               + "\n bias:" + str(model.intercept_),loc='left')
#     ax.set_xlabel('log distance (m)')
#     ax.set_ylabel('pathloss (dB)')

#     plt.show()
    
def lassoRegression(X_train, y_train, X_val, y_val, X_test, y_test, label="test"):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    return model
#     train_prediction = model.predict(X_train)
#     train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));

#     val_prediction = model.predict(X_val)
#     val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));

#     test_prediction = model.predict(X_test)
#     test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));

#     plt.figure(figsize=(12, 5))
#     ax = plt.axes()
#     ax.scatter(X_train, y_train)
#     ax.plot(X_train, train_prediction, color="red")
#     plt.title("<" + label + ">\nRMSE(train) =" + str(train_rmse)
#               + "\nRMSE(val) =" + str(val_rmse)
#               + "\nRMSE(test) =" + str(test_rmse)
#               + "\ncoefficient:" + str(model.coef_)
#               + "\nbias:" + str(model.intercept_),loc='left' )
#     ax.set_xlabel('log distance (m)')
#     ax.set_ylabel('pathloss (dB)')

#     plt.show()
    
def polynomialRegression(X_train, y_train, X_val, y_val, X_test, y_test, ndeg=3, label="test"):
    polynomial_features= PolynomialFeatures(degree=ndeg)
    
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_val_poly = polynomial_features.fit_transform(X_val)
    X_test_poly = polynomial_features.fit_transform(X_test)
    model = Pipeline([('poly', PolynomialFeatures(degree=ndeg)),('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(X_train_poly, y_train)

    return model, X_train_poly, X_val_poly, X_test_poly  
    
#     train_prediction = model.predict(X_train_poly)
#     train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));

#     X_val_poly = polynomial_features.fit_transform(X_val)
#     val_prediction = model.predict(X_val_poly)
#     val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));

#     X_test_poly = polynomial_features.fit_transform(X_test)
#     test_prediction = model.predict(X_test_poly)
#     test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));

    # print("coefficient:" + str(model.coef_))
    # print("bias:" + str(model.intercept_))

#     plt.figure(figsize=(12, 5))
#     ax = plt.axes()
#     ax.scatter(X_train, y_train)
#     ax.plot(X_train, train_prediction, color="red")
#     plt.title("RMSE(train) =" + str(train_rmse)
#               + "\nRMSE(val) =" + str(val_rmse)
#               + "\nRMSE(test) =" + str(test_rmse), loc="left")

#     ax.set_xlabel('log distance (m)')
#     ax.set_ylabel('pathloss (dB)')

#     plt.show()
