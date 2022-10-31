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
                 dilations: List[int] = [1, 2, 4, 8],
                 kernel_initializer: str = "he_normal",
                 dropout_rate: float = 0,
                 use_weight_norm: bool = False,
                 use_attention: bool = False,
                 **kwargs):

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.use_weight_norm = use_weight_norm
        self.use_attention = use_attention
        super(PrototypeLayerII, self).__init__(**kwargs)

    def build(self, input_shape):


        super(PrototypeLayerII, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input, **kwargs):


        return None
