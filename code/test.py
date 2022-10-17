import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import basename
import torch.nn as nn
from d2l import torch as d2l
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import *
from keras import initializers
from keras.models import Model
import graphviz
from sklearn.metrics import *
import math


# x_test = np.array([[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]])
# X = Input(shape=(6, 2))
# Y = Lambda(lambda x: x[:, -1, :], output_shape=(3, ))(X)

# model = Model(X, Y)
# print(model.output_shape)
# print(model.predict(x_test))
# print(x_test.shape)

# a = np.arange(6).reshape((1, 3, 2))
# b = np.arange(6, 12).reshape((1, 3, 2))
# # b = np.arange(20).reshape((2, 5, 2))
# c = Concatenate(axis=0)([a, b])
# print(c)

import random
random.seed(3)

Y = np.array([1, 2, 3, 4, 5])
Y_ = np.array([3, 3, 3, 3, 3])

mse = mean_squared_error(Y, Y_)  # 均方误差
rmse = math.sqrt(mse)  # 均方根误差
sse = len(Y) * mse  # 和方差
mae = mean_absolute_error(Y, Y_)  # 平均绝对误差
mape = mean_absolute_percentage_error(Y, Y_)
r2 = r2_score(Y, Y_)  # r2分数（越接近+1越好）

print("MSE = {:.4f}".format(mse))
print("RMSE = {:.4f}".format(rmse))
print("SSE = {:.4f}".format(sse))
print("MAE = {:.4f}".format(mae))
print("MAPE = {:.4f}".format(mape))
print("R2 = {:.4f}".format(r2))

# t = torch.rand(1680)
# t = torch.reshape(t, (1, 1, 10, 168))
# d2l.show_heatmaps(t, 'x', 'y')
# plt.show()


