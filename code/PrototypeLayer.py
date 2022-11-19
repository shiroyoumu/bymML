import random
from typing import List
from tensorflow_addons.layers import WeightNormalization
from keras import backend as K, Input, Model
from keras.layers import Layer, Conv1D, Dropout, Add, Concatenate, Lambda
import numpy as np
# from attention import Attention
import tensorflow as tf


class PrototypeLayer(Layer):
    def __init__(self,
                 filters: int = 1,
                 kernel_size: int = 3,
                 dilations: List[int] = [1, 2, 4, 8],
                 kernel_initializer: str = "he_normal",
                 dropout_rate: float = 0,
                 use_weight_norm: bool = False,
                 use_attention: bool = False,
                 return_sequences: bool = False,
                 **kwargs):

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.use_weight_norm = use_weight_norm
        self.use_attention = use_attention
        self.return_sequences = return_sequences
        self.medLayers = []
        self.dropoutLayer = None
        self.sliceLayer = None
        self.addLayer = None
        self.concatLayer = None
        self.attentionLayer = None
        super(PrototypeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.medLayers = []
        for d in self.dilations:
            layer = Conv1D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           dilation_rate=d,
                           padding="causal",
                           kernel_initializer=self.kernel_initializer,
                           use_bias=False)
            if self.use_weight_norm:
                layer = WeightNormalization(layer)
            self.medLayers.append(layer)
            layer = Conv1D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           dilation_rate=d,
                           padding="causal",
                           kernel_initializer=self.kernel_initializer,
                           use_bias=False)
            if self.use_weight_norm:
                layer = WeightNormalization(layer)
            self.medLayers.append(layer)

        self.dropoutLayer = Dropout(self.dropout_rate)
        self.sliceLayer = Lambda(lambda x: x[:, -1, :])
        self.addLayer = Add()
        self.concatLayer = Concatenate(axis=-1)
        self.attentionLayer = Attention(1)
        super(PrototypeLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input, **kwargs):
        layersOutputs = []
        x = z = input
        for i, L in enumerate(self.medLayers):
            temp = L(x)
            temp = K.relu(temp)
            x = self.dropoutLayer(temp)
            if i % 2 == 1:
                if self.return_sequences:
                    layersOutputs.append(x)
                else:
                    layersOutputs.append(self.sliceLayer(x))
                x = z = K.relu(z + x)

        if (self.return_sequences is False) or (not self.use_attention): # 不用Attention
            return self.addLayer(layersOutputs)
        else:   # 用Attention, return_sequences==True才有意义
            attenInput = self.concatLayer(layersOutputs)
            attenInput = tf.transpose(attenInput, [0, 2, 1])
            return self.attentionLayer(attenInput)
