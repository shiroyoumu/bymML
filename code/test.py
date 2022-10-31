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
from PrototypeLayer import PrototypeLayer
from keras import backend as K

# x = np.arange(10).astype("float32").reshape([1, 5, 2])
# x = tf.convert_to_tensor(x)
# print(x)
#
#
# y = tf.transpose(x, [0, 2, 1])
#
# print(y)


x = np.arange(10).astype("float32").reshape([1, 10, 1]) - 4.5
print(x)

y = PrototypeLayer(filters=2,
                   kernel_size=3,
                   dilations=[1, 2, 4],
                   dropout_rate=0,
                   use_weight_norm=False,
                   kernel_initializer="Ones",
                   return_sequences=True,
                   use_attention=True)(x)
print(y)

z = TCN(nb_filters=2, kernel_size=3, dilations=[1, 2, 4], kernel_initializer="Ones", return_sequences=False)(x)
print(z)







# a = np.arange(10).astype("float32").reshape([1, 5, 2])
# # b = Conv1D(filters=3, kernel_size=3, kernel_initializer="Ones")(a)
# b = TCN(nb_filters=2, kernel_size=3, dilations=[1], kernel_initializer="Ones", return_sequences=True)(a)
# b = Attention(1)(b)
# print(b)


# a = np.arange(3).astype("float32").reshape([1, 3])
# c = TCN(nb_filters=3, kernel_size=3, dropout_rate=0.1, dilations=(1, 2, 4, 8))(a)
# c = MyLayer(3)(a)

# x = np.arange(12).astype("float32").reshape([1, 6, 2])
#
# y1 = Dense(int(x.shape[-1]), use_bias=False, name='attention_score_vec', kernel_initializer="Ones")(x)
# y2 = Lambda(lambda x: x[:, -1, :], output_shape=(int(x.shape[-1]), ))(x)
# y = Dot(axes=[1, 2])([y2, y1])
# print(y)


# print(c)
# x = Attention(4)(a)
None




# t = torch.rand(1680)
# t = torch.reshape(t, (1, 1, 10, 168))
# d2l.show_heatmaps(t, 'x', 'y')
# plt.show()


