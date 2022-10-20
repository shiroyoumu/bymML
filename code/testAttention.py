def build(self, input_shape):
    # member to hold current output shape of the layer for building purposes
    self.build_output_shape = input_shape

    # list to hold all the member ResidualBlocks
    self.residual_blocks = []
    total_num_blocks = len(self.dilations)
    if not self.use_skip_connections:
        total_num_blocks += 1  # cheap way to do a false case for below


    for i, d in enumerate(self.dilations):
        # 过滤器数量可以是list，代表每层卷积使用不同的过滤器数量
        res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
        self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                  nb_filters=res_block_filters,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  activation=self.activation_name,
                                                  dropout_rate=self.dropout_rate,
                                                  use_batch_norm=self.use_batch_norm,
                                                  use_layer_norm=self.use_layer_norm,
                                                  use_weight_norm=self.use_weight_norm,
                                                  kernel_initializer=self.kernel_initializer,
                                                  name='residual_block_{}'.format(len(self.residual_blocks))))
        # build newest residual block
        self.residual_blocks[-1].build(self.build_output_shape)
        self.build_output_shape = self.residual_blocks[-1].res_output_shape

    # this is done to force keras to add the layers in the list to self._layers
    # for layer in self.residual_blocks:
    #     self.__setattr__(layer.name, layer)

    self.slicer_layer = Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')
    self.slicer_layer.build(self.build_output_shape.as_list())

def call(self, inputs, training=None, **kwargs):
    x = inputs

    if self.go_backwards:
        # reverse x in the time axis
        x = tf.reverse(x, axis=[1])

    self.layers_outputs = [x]
    self.skip_connections = []
    for res_block in self.residual_blocks:
        try:
            x, skip_out = res_block(x, training=training)
        except TypeError:  # compatibility with tensorflow 1.x
            x, skip_out = res_block(K.cast(x, 'float32'), training=training)
        self.skip_connections.append(skip_out)
        self.layers_outputs.append(x)

    if self.use_skip_connections:
        if len(self.skip_connections) > 1:
            # Keras: A merge layer should be called on a list of at least 2 inputs. Got 1 input.
            x = layers.add(self.skip_connections, name='Add_Skip_Connections')
        else:
            x = self.skip_connections[0]
        self.layers_outputs.append(x)

    if not self.return_sequences:
        # case: time dimension is unknown. e.g. (bs, None, input_dim).
        if self.padding_same_and_time_dim_unknown:
            self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
        x = self.slicer_layer(x)
        self.layers_outputs.append(x)
    return x