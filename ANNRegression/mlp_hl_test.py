import matplotlib.pyplot as plt
import numpy as np
import json

from MLP.mlp_regression import validation_test

validation_test('hl', 11, 50, 'tanh', 'lbfgs')
validation_test('hl', 11, 50, 'relu', 'lbfgs')
validation_test('hl', 11, 50, 'logistic', 'lbfgs')
validation_test('hl', 11, 50, 'identity', 'lbfgs')
