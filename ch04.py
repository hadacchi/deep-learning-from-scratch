import numpy as np


def cross_entropy_error(y, t):
    """t is one hot true label
    y is output of neural network
    t and y have k-dimensional value, where k is the number of label
    """

    EPS = 1.0e-10
    t = np.array(t)
    y = np.array(y)

    return -(t * np.log(y + EPS)).sum()
