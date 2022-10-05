import numpy
import numpy as np

a = numpy.array([[1,],
               [4,],
               [7,]])
b = numpy.array([[2,],
               [5,],
               [8,]])

print(np.append(a, b, axis=0))