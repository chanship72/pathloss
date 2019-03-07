import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from LIN_REG.data_loader import data_loader_pathloss

X_train, y_train, X_val, y_val, X_test, y_test = data_loader_pathloss('PLdata.mat')

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
train_rmse = np.sqrt(np.mean((y_train-train_prediction)**2));
print(train_rmse)

val_prediction = model.predict(X_val)
val_rmse = np.sqrt(np.mean((y_val-val_prediction)**2));
print(val_rmse)

test_prediction = model.predict(X_test)
test_rmse = np.sqrt(np.mean((y_test-test_prediction)**2));
print(test_rmse)

plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(X_train, y_train)
ax.plot(X_train, train_prediction, color="red")
plt.title("RMSE(train) =" + str(train_rmse)
          + "\n RMSE(val) =" + str(val_rmse)
          + "\n RMSE(test) =" + str(test_rmse)
          + "\n coefficient:" + str(model.coef_)
          + "\n bias:" + str(model.intercept_))
ax.set_xlabel('log distance (m)')
ax.set_ylabel('pathloss (dB)')

plt.show()