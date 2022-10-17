import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

import sqlite3 as lite

hosts = ['host0001', 'host0021', 'host0027', 'host0029']
con = lite.connect("../data/dataset_db.db")
df = pd.read_sql("select Mean from datasetDB where hostname in {}".format(str(tuple(hosts))), con)

None




# t = torch.rand(1680)
# t = torch.reshape(t, (1, 1, 10, 168))
# d2l.show_heatmaps(t, 'x', 'y')
# plt.show()


