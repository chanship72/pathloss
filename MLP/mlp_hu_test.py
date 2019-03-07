import matplotlib.pyplot as plt
import numpy as np
import json

from mlp_regression import validation_test

validation_test('hu', 1, 302, 'tanh', 'lbfgs')
validation_test('hu', 1, 302, 'relu', 'lbfgs')
validation_test('hu', 1, 302, 'logistic', 'lbfgs')
validation_test('hu', 1, 302, 'identity', 'lbfgs')