from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from LIN_REG.data_loader import data_loader_pathloss

import numpy as np
import matplotlib.pyplot as plt

m = 50
X_train, y_train, X_val, y_val, X_test, y_test = data_loader_pathloss('PLdata.mat')

polynomial_features= PolynomialFeatures(degree=m)
X_train_poly = polynomial_features.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_poly, y_train)

train_prediction = model.predict(X_train_poly)
train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));
print(train_rmse)

X_val_poly = polynomial_features.fit_transform(X_val)
val_prediction = model.predict(X_val_poly)
val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));
print(val_rmse)

X_test_poly = polynomial_features.fit_transform(X_test)
test_prediction = model.predict(X_test_poly)
test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));
print(test_rmse)

print("coefficient:" + str(model.coef_))
print("bias:" + str(model.intercept_))

plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(X_train, y_train)
ax.plot(X_train, train_prediction, color="red")
plt.title("RMSE(train) =" + str(train_rmse)
          + "\nRMSE(val) =" + str(val_rmse)
          + "\nRMSE(test) =" + str(test_rmse)
          + "\ncoefficient:" + str(model.coef_)
          + "\nbias:" + str(model.intercept_),loc="left")

ax.set_xlabel('log distance (m)')
ax.set_ylabel('pathloss (dB)')

plt.show()