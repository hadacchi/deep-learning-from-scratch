import numpy as np


import matplotlib.pyplot as plt


# activation function
def step(x):
    if hasattr(x, "__len__"):
        return np.array([1 if e > 0 else 0 for e in x])
    return 1 if x > 0 else 0


def sigmoid(x):
    if hasattr(x, "__len__"):
        return np.array([1 / (1 + np.exp(-e)) for e in x])
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# identity function
def softmax(x):
    """softmax function is defined as:
    y_k = \\frac{exp(a_k)}{\\sum_i exp(a_i)}

    then, large a_i cause overflow.
    now,

    y_k = \\frac{C exp(a_k)}{C \\sum_i exp(a_i)}
        = \\frac{exp(a_k + log C)}{\\sum_i exp(a_i + log C)}

    let C' is log C,

    y_k = \\frac{exp(a_k + C')}{\\sum_i exp(a_i + C')}

    if C' is -max(a_i), a_k+C' do not cause overflow.

    Example
    -------
    >>> a = np.array([ .3, 2.9, 4 ])
    >>> softmax(a)
    array([0.01821127, 0.24519181, 0.73659691])
    >>> b = np.array([ 1010, 1000, 990 ])
    >>> softmax(b)
    array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
    """

    # this ipementation may cause buffer overflow
    # return np.exp(x)/np.exp(x).sum()

    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


# show function form
def plot_function(h, title):
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, h(x))
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close("all")


# forward
def forward(network, x, h):
    """forward neural network

    Parameters
    ----------
    network : list of neural network's weights and bias
        network['w'] is list of weights
        network['b'] is list of biases
    x : np.ndarray
        input
    h : activation function and identity function
        identity function is the last element

    Example
    -------
    >>> X = np.array([ 1, .5 ])
    >>> W1 = np.array([[ .1, .3, .5 ], [ .2, .4, .6 ]])
    >>> B1 = np.array([ .1, .2, .3 ])
    >>> W2 = np.array([[ .1, .4 ], [ .2, .5 ], [ .3, .6 ]])
    >>> B2 = np.array([ .1, .2 ])
    >>> W3 = np.array([[ .1, .3 ], [ .2, .4 ]])
    >>> B3 = np.array([ .1, .2 ])
    >>> network = { 'w': [W1, W2, W3], 'b': [B1, B2, B3] }
    >>> h = [sigmoid, sigmoid, lambda x: x]
    >>> forward(network, X, h)
    array([0.31682708, 0.69627909])
    """

    ws = network["w"]
    bs = network["b"]

    z = X

    for w, b, func in zip(ws, bs, h):
        a = np.dot(z, w) + b
        z = func(a)

    return z


# 以下のニューラルネットワークを実装する
# 入力層 2  x1, x2
# バイアス 1 b1
# 隠れ層1 … 3 ノード a1_1, a1_2, a1_3
# 隠れ層2 … 2 ノード a2_1, a2_2
# 出力層 … 2ノード y1, y2

x1 = 1
x2 = 0.5

w1_11 = 0.1
w1_21 = 0.3
w1_31 = 0.5

w1_12 = 0.2
w1_22 = 0.4
w1_32 = 0.6

b1_1 = 0.1
b1_2 = 0.2
b1_3 = 0.3

h1 = sigmoid

X = np.array([x1, x2])
W1 = np.array([[w1_11, w1_21, w1_31], [w1_12, w1_22, w1_32]])
B1 = np.array([b1_1, b1_2, b1_3])
A1 = np.dot(X, W1) + B1
Z1 = h1(A1)

# a1_1 = w1_11 * x1 + w1_12 * x2 + b1_1
# a1_2 = w1_21 * x1 + w1_22 * x2 + b1_2
# a1_3 = w1_31 * x1 + w1_32 * x2 + b1_3
#
# z1_1 = h1(a1_1)
# z1_2 = h1(a1_2)
# z1_3 = h1(a1_3)

# print(f'z1_1 = {z1_1}, z1_2 = {z1_2}, z1_3 = {z1_3}')
# print(f'Z1 = \n{Z1}')

w2_11 = 0.1
w2_21 = 0.4

w2_12 = 0.2
w2_22 = 0.5

w2_13 = 0.3
w2_23 = 0.6

b2_1 = 0.1
b2_2 = 0.2

h2 = sigmoid

W2 = np.array([[w2_11, w2_21], [w2_12, w2_22], [w2_13, w2_23]])
B2 = np.array([b2_1, b2_2])
A2 = np.dot(Z1, W2) + B2
Z2 = h2(A2)

# a2_1 = w2_11 * z1_1 + w2_12 * z1_2 + w2_13 * z1_3 + b2_1
# a2_2 = w2_21 * z1_1 + w2_22 * z1_2 + w2_23 * z1_3 + b2_2
#
# z2_1 = h2(a2_1)
# z2_2 = h2(a2_2)

# print(f'z2_1 = {z2_1}, z2_2 = {z2_2}')
# print(f'Z2 = \n{Z2}')

w3_11 = 0.1
w3_21 = 0.3

w3_12 = 0.2
w3_22 = 0.4

b3_1 = 0.1
b3_2 = 0.2


def out(x):
    return x


W3 = np.array([[w3_11, w3_21], [w3_12, w3_22]])
B3 = np.array([b3_1, b3_2])
A3 = np.dot(Z2, W3) + B3
Y = out(A3)

# a3_1 = w3_11 * z2_1 + w3_12 * z2_2 + b3_1
# a3_2 = w3_21 * z2_1 + w3_22 * z2_2 + b3_2
#
# y1 = out(a3_1)
# y2 = out(a3_2)

# print(f'y1 = {y1}, y2 = {y2}')
# print(f'Y = \n{Y}')


if __name__ == "__main__":
    # output function response for debugging

    # print (y1)
    # print (y2)
    # print(Y)
    import doctest

    doctest.testmod()
