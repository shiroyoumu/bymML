import random
from typing import List
from tensorflow_addons.layers import WeightNormalization
from keras import backend as K, Input, Model
from keras.layers import Layer, Conv1D, Dropout, Add, Concatenate, Lambda
import numpy as np
from attention import Attention
import tensorflow as tf


class PrototypeLayerII(Layer):
    def __init__(self,
                 filters: int = 1,
                 kernel_size: int = 3,
                 dilation_rate: int = 1,
                 kernel_initializer: str = "he_normal",
                 use_attention: bool = False,
                 **kwargs):

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.use_attention = use_attention

        super(PrototypeLayerII, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernels = self.add_weight(name='kernels',
                                       shape=(self.filters, self.kernel_size, input_shape[2]),  # input_shape[2]
                                       initializer=self.kernel_initializer,
                                       trainable=True)

        super(PrototypeLayerII, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input, **kwargs):
        results = []
        for k in range(self.kernels.shape[0]):
            y = self.DoMul(input, self.kernels[k, :, :], self.dilation_rate)
            results.append(y)
        return K.concatenate(results, axis=-1)

    def DoMul(self, input, k, d):
        '''
        计算一次1d卷积

        :param input: 输入序列（batch_size, time_step, depth）
        :param k: 卷积核（kernel_size, depth）
        :param d: 间隔
        :return: 卷积结果
        '''
        time = input.shape[1]
        depth = input.shape[2]
        k_size = k.shape[0]
        addLines = (k_size - 1) * d
        z = np.zeros(addLines * depth).astype("float32").reshape([1, addLines, depth])
        z_input = K.concatenate([z, input], axis=1)
        result = []
        for i in range(time):
            y = 0
            for j in range(k_size):
                y += np.vdot(k[j, :], z_input[:, i - d * (k_size - j - 1) + addLines, :])
            result.append(y)
        result = np.array(result).astype("float32").reshape([1, len(result), 1])
        return tf.convert_to_tensor(result)



























