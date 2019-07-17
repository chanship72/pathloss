import matplotlib.pyplot as plt
import numpy as np

from dataload import data_processing_linear_regression

x, y = data_processing_linear_regression('PLdata.mat')
n = len(x)
w = np.random.normal(0, 0.1)
b = np.random.normal(0, 0.1)

# print(w,b)
# size eta
max_iter = 100000   # iteration
lr = 0.001  #learning rate
for t in range(0, max_iter):
    grad_t = 0.
    grad_b = 0.

    h = (np.multiply(w, x) + b) - y
    grad_t += 2*np.dot(x.T,h)/n
    grad_b += 2*np.sum(h)/n

    # Update the weights
    w = w - lr*grad_t
    b = b - lr*grad_b

# Plot the data and best fit line
yfit = np.multiply(w,x) + b
rmse = np.sqrt(np.mean((y-yfit)**2));
print(rmse)
plt.scatter(x, y)
plt.plot(x, yfit, color="red")
plt.xlabel('log distance (m)')
plt.ylabel('pathloss (dB)')
plt.title('root mean square error =' + str(rmse) + " coefficient:"
          + str(w) + " bias:" + str(b))

plt.show()