from tcn import TCN, tcn_full_summary
from keras.layers import Dense
from keras.models import Sequential

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, 20, 1


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, time_steps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0  # we introduce the target in the first timestep of the sequence.
    y_train[pos_indices, 0] = 1.0  # the task is to see if the TCN can go back in time to find it.
    return x_train, y_train


tcn_layer = TCN(input_shape=(time_steps, input_dim))
# The receptive field tells you how far the model can see in terms of timesteps.
print('Receptive field size =', tcn_layer.receptive_field)

m = Sequential([
    tcn_layer,
    Dense(1)
])

m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)
print(x, y)