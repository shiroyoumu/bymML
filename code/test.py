import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from os.path import basename
import torch.nn as nn
from d2l import torch as d2l

t = torch.rand(1680)
t = torch.reshape(t, (1, 1, 10, 168))

d2l.show_heatmaps(t, 'x', 'y')
plt.show()


