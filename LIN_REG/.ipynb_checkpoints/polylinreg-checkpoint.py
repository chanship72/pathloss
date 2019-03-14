from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from LIN_REG.data_loader import data_loader_pathloss

import numpy as np
import matplotlib.pyplot as plt

def polynomialRegression(X_train, y_train, X_val, y_val, X_test, y_test, ndeg=50, label="test"):
    polynomial_features= PolynomialFeatures(degree=ndeg)
    
    X_train_poly = polynomial_features.fit_transform(X_train)
    model = Pipeline([('poly', PolynomialFeatures(degree=ndeg)),('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(X_train_poly, y_train)
    train_prediction = model.predict(X_train_poly)
    train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));

    X_val_poly = polynomial_features.fit_transform(X_val)
    val_prediction = model.predict(X_val_poly)
    val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));

    X_test_poly = polynomial_features.fit_transform(X_test)
    test_prediction = model.predict(X_test_poly)
    test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));

    # print("coefficient:" + str(model.coef_))
    # print("bias:" + str(model.intercept_))

    plt.figure(figsize=(12, 5))
    ax = plt.axes()
    ax.scatter(X_train, y_train)
    ax.plot(X_train, train_prediction, color="red")
    plt.title("RMSE(train) =" + str(train_rmse)
              + "\nRMSE(val) =" + str(val_rmse)
              + "\nRMSE(test) =" + str(test_rmse), loc="left")

    ax.set_xlabel('log distance (m)')
    ax.set_ylabel('pathloss (dB)')

    plt.show()
