import numpy as np
from dataset.mnist import load_mnist
from ch03_3 import init_network, predict


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
