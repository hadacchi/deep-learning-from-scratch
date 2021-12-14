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


# show function form
def plot_function(h, title):
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, h(x))
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close("all")


# X           W         Y
#
# [ x1 ] ---- w11 ----> [ y1 ]
# [ x1 ] ---- w21 ----> [ y2 ]
# [ x1 ] ---- w31 ----> [ y3 ]
#
# [ x2 ] ---- w12 ----> [ y1 ]
# [ x2 ] ---- w22 ----> [ y2 ]
# [ x2 ] ---- w32 ----> [ y3 ]
#

x1 = 3
x2 = 7

w11 = 0.1
w21 = 0.5
w31 = 0.3

w12 = 0.3
w22 = 0.2
w32 = 0.7

X = np.array([x1, x2])

W = np.array([[w11, w21, w31], [w12, w22, w32]])


if __name__ == "__main__":
    # output function response for debugging

    print("step function")
    plot_function(step, "step function")
    print("sigmoid function")
    plot_function(sigmoid, "sigmoid function")
    print("ReLU function")
    plot_function(relu, "ReLU function")

    print(f"X = \n{X}")
    print(f"W = \n{W}")
    print(f"Y = \n{np.dot(X,W)}")
