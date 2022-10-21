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


# a = np.arange(3).astype("float32").reshape([1, 3])
# c = TCN(nb_filters=3, kernel_size=3, dropout_rate=0.1, dilations=(1, 2, 4, 8))(a)
# c = MyLayer(3)(a)

x = np.arange(12).astype("float32").reshape([1, 6, 2])

y1 = Dense(int(x.shape[-1]), use_bias=False, name='attention_score_vec', kernel_initializer="Ones")(x)
y2 = Lambda(lambda x: x[:, -1, :], output_shape=(int(x.shape[-1]), ))(x)
y = Dot(axes=[1, 2])([y2, y1])
print(y)


# print(c)
# x = Attention(4)(a)
None




# t = torch.rand(1680)
# t = torch.reshape(t, (1, 1, 10, 168))
# d2l.show_heatmaps(t, 'x', 'y')
# plt.show()


