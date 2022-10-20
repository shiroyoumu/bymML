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
from attention import Attention
from tcn import TCN, tcn_full_summary
import tensorflow as tf

x = np.arange(10).astype("float32").reshape([1, 10, 1])
y = x
y = Conv1D(filters=1, kernel_size=3, dilation_rate=1, padding="causal", kernel_initializer="Ones")(y)
y = Conv1D(filters=1, kernel_size=3, dilation_rate=1, padding="causal", kernel_initializer="Ones")(y)
z1 = y
y = w = y + x
y = Conv1D(filters=1, kernel_size=3, dilation_rate=2, padding="causal", kernel_initializer="Ones")(y)
y = Conv1D(filters=1, kernel_size=3, dilation_rate=2, padding="causal", kernel_initializer="Ones")(y)
z2 = y
y = y + w
y = Conv1D(filters=1, kernel_size=3, dilation_rate=4, padding="causal", kernel_initializer="Ones")(y)
y = Conv1D(filters=1, kernel_size=3, dilation_rate=4, padding="causal", kernel_initializer="Ones")(y)
z3 = y

print(z1 + z2 + z3)

z = TCN(nb_filters=1,
        kernel_size=3,
        dilations=[1, 2, 4],
        kernel_initializer="Ones",
        return_sequences=True)(x)

print(z)


# x = np.arange(10).astype("float32").reshape([1, 10, 1])
# y = x
# y = Conv1D(filters=1, kernel_size=3, dilation_rate=1, padding="causal", kernel_initializer="Ones")(y)
# y = Conv1D(filters=1, kernel_size=3, dilation_rate=1, padding="causal", kernel_initializer="Ones")(y)
#
# print(y + x)














































