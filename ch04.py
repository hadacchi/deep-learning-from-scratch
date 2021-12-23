import numpy as np
from dataset.mnist import load_mnist
from ch03_3 import init_network, predict
import matplotlib.pyplot as plt


def cross_entropy_error(y, t):
    """t is one hot true label
    y is output of neural network
    t and y have k-dimensional value, where k is the number of label
    """

    EPS = 1.0e-10
    t = np.array(t)
    y = np.array(y)

    return -(t * np.log(y + EPS)).sum()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

# print(x_train.shape)
# print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 100
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# print(x_batch)
# print(t_batch)

network = init_network()
y_batch = predict(network, x_batch)
p_batch = np.argmax(y_batch, axis=1)
print(p_batch)
print(p_batch == t_batch)

print(cross_entropy_error(p_batch, t_batch) / len(p_batch))

# different


def func1(x):
    """0.01x^2+0.1x"""
    return (0.1 * x + 1) * 0.1 * x


def numerical_diff(func, x):
    """df/dx(x)"""

    # hが小さすぎると，誤差の原因となる
    h = 1.0e-4
    return (func(x + h) - func(x - h)) / h / 2


x = np.arange(-2, 20, 0.1)
f = np.array([func1(s) for s in x])
plt.plot(x, f)

x1 = 5
dfdx = numerical_diff(func1, x1)
# y-func1(x1) = dfdx * (x-x1)
# y = dfdx*x - x1*dfdx + func1(x1)
C = func1(x1) - x1 * dfdx
# 接線
y = dfdx * x + C
plt.plot(x, y)
plt.title(f"$f(x)=0.01x^2+0.1x, df/dx({x1})={dfdx:.2f}$")
plt.savefig("ch04.png")
plt.close("all")

# gradient


def func2(x):
    """$x_0^2+x_1^2$"""
    return x[0] * x[0] + x[1] * x[1]


def part_diff(func, xs, axis=0):
    """$\\round f/\\round x"""

    EPS = 1.0e-4
    x = np.zeros_like(xs)
    x[axis] = xs[axis]
    h = np.zeros_like(xs)
    h[axis] = EPS
    print(x + h, x - h)

    return (func(x + h) - func(x - h)) / EPS / 2


def gradient(func, xs):
    return np.array([part_diff(func, xs, axis=i) for i in range(len(xs))])


# print(part_diff(func2, np.array([3.0, 4.0]), axis=0))
# print(part_diff(func2, np.array([3.0, 4.0]), axis=1))
print(gradient(func2, np.array([3.0, 4.0])))

x = np.arange(-2, 2.1, 0.5)
y = np.arange(-2, 2.1, 0.5)

X, Y = np.array(np.meshgrid(x, y)).T.reshape(2, len(x) * len(x))

X = []
Y = []
U = []
V = []
for xx in x:
    for yy in y:
        X.append(xx)
        Y.append(yy)
        u, v = -1 * gradient(func2, np.array([xx, yy]))
        U.append(u)
        V.append(v)
U = np.array(U)
V = np.array(V)

print(U, V)

plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=10)
plt.show()
