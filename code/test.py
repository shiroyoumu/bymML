import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import torch
from os.path import basename
# import torch.nn as nn
# from d2l import torch as d2l
import os
import sqlite3 as lite
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import *
from keras import initializers
from keras.models import Model
# import graphviz
from matplotlib import font_manager
from sklearn.metrics import *
import math
# from attention import Attention
from tcn import TCN, tcn_full_summary
import tensorflow as tf

from keras import backend as K
import xgboost
from xgboost import XGBClassifier
from functions import *


a = np.arange(20).astype("float32").reshape([1, 20, 1])
# b = Lambda(lambda x: x[:, -7:, :])(a)


a = np.full((3, 3), np.nan)
print(a)



# import numpy as np
# a=np.array([[1,np.nan,3,4],[np.nan,np.nan,5,6]]);
# a[np.where(np.isnan(a))]=777
#
# print(a)





# path = "../data/l.csv"
# data = pd.read_csv(path)
# list = list(data['Unnamed: 0'])
# print(len(list))
# list2 = []
# for i in list:
#     list2.append("" + i)
# print(len(list2))
# print(list2)
#
# None










# font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
#
# a = np.array([1,2,80,2,3,1,4,50,10,3,
#               1,2,5,60,12,3,1,2,12,24,
#               30,5,9,21,35,19,13]).reshape((27, 1))
#
#
# plt.plot(a, label='原始序列')
#
# # plt.plot(a, label='a')
# b = SmoothSet(a, 2, 20)
# plt.plot(b, label='平滑后序列')
# # plt.plot(b, label='b')
# plt.xlabel('时间步长',fontproperties='SimHei')
# plt.ylabel('指标数值',fontproperties='SimHei')
#
# plt.legend(prop=font)
# plt.show()

#
# hosts = ['host0001', 'host0021', 'host0354']
#
#
# pathDataDB = "../data/dataset_db.db"    # 数据库文件
# con = lite.connect(pathDataDB)
#
# data = CollectTrainData(con, ['host0001', 'host0021'])
# plt.plot(data)
# b = SmoothSet(data, 1, 20)
#
# plt.plot(b)
# plt.show()















# import xgboost
# # First XGBoost model for Pima Indians dataset
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # load data
# dataset = loadtxt('../data/pima-indians-diabetes.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # split data into train and test sets
# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# # fit model no training data
# model = XGBClassifier()
# model.fit(X_train, y_train)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))



# x = np.arange(5).astype("float32").reshape([1, 5, 1])
#
# y = PrototypeLayer(filters=3,
#                    kernel_size=3,
#                    dilations=[1, 2, 4, 8, 16, 32],
#                    return_sequences=True,
#                    use_attention=True,
#                    dropout_rate=0.4,
#                    use_weight_norm=True)(x)
# print(y)

# def DoMul(input, k, d):
#     time = input.shape[1]
#     depth = input.shape[2]
#     k_size = k.shape[0]
#     addLines = (k_size - 1) * d
#     z = np.zeros(addLines * depth).astype("float32").reshape([1, addLines, depth])
#     z_input = K.concatenate([z, input], axis=1)
#     result = []
#     for i in range(time):
#         y = 0
#         for j in range(k_size):
#             y += np.vdot(k[j, :], z_input[:, i - d * (k_size - j - 1) + addLines, :])
#         result.append(y)
#     result = np.array(result).reshape([1, len(result), 1])
#     return tf.convert_to_tensor(result)

# x = np.arange(5).astype("float32").reshape([1, 5, 1])
# x = tf.convert_to_tensor(x)
# print(x)
#
# print(tf.multiply(x, x))
# print(np.vdot(x, x))



# k = np.ones(6).astype("float32").reshape([3, 2])
# k = tf.convert_to_tensor(k)
# print(k)

# y = PrototypeLayerII(filters=1, kernel_size=3, dilations=[1, 2, 4, 8], kernel_initializer="Ones")(x)
# print(y)
#
# y2 = TCN(nb_filters=1, kernel_size=3, dilations=[1, 2, 4, 8], kernel_initializer="Ones", return_sequences=True)(x)
# print(y2)

#
# y1 = Conv1D(filters=3, kernel_size=4, padding="causal", dilation_rate=2, kernel_initializer="Ones")(x)
# y2 = PrototypeLayerIII(filters=3, kernel_size=4, dilation_rate=2, kernel_initializer="Ones")(x)
# print(y1)
# print(y2)









# y = tf.transpose(x, [0, 2, 1])
#
# print(y)


# x = np.arange(10).astype("float32").reshape([1, 10, 1]) - 4.5
# print(x)
#
# y = PrototypeLayer(filters=2,
#                    kernel_size=3,
#                    dilations=[1, 2, 4],
#                    dropout_rate=0,
#                    use_weight_norm=False,
#                    kernel_initializer="Ones",
#                    return_sequences=True,
#                    use_attention=True)(x)
# print(y)
#
# z = TCN(nb_filters=2, kernel_size=3, dilations=[1, 2, 4], kernel_initializer="Ones", return_sequences=False)(x)
# print(z)
#
# tf.nn.convolution



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


# t = torch.rand(1680)
# t = torch.reshape(t, (1, 1, 10, 168))
# d2l.show_heatmaps(t, 'x', 'y')
# plt.show()


