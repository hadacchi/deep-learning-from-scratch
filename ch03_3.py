from dataset.mnist import load_mnist
import numpy as np
import pickle
import time


def sigmoid(x):
    if hasattr(x, "__len__"):
        return np.array([1 / (1 + np.exp(-e)) for e in x])
    return 1 / (1 + np.exp(-x))


# identity function
def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def mnist_test():
    from PIL import Image

    # download mnist data
    train, _ = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    # train, test are (image, label)
    img = train[0][0]
    img = img.reshape(28, 28)
    Image.fromarray(np.uint8(img)).show()


def get_test_data():
    _, test = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return test


def init_network():
    with open("ch03/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # hidden layer
    ws = [k for k in sorted(network.keys()) if k[0] == "W"]
    Ws = [network[k] for k in ws[:-1]]

    bs = [k for k in sorted(network.keys()) if k[0] == "b"]
    Bs = [network[k] for k in bs[:-1]]

    z = x

    for weight, bias in zip(Ws, Bs):
        a = np.dot(z, weight) + bias
        z = sigmoid(a)

    # output layer
    W = network[ws[-1]]
    B = network[bs[-1]]

    a = np.dot(z, W) + B
    y = softmax(a)

    return y


if __name__ == "__main__":
    # from PIL import Image

    # mnist_test()
    network = init_network()

    test_X, test_t = get_test_data()
    counter = 0
    # without batch
    # for x, t in zip(test_X, test_t):
    #    y = predict(network, x)
    #    p = np.argmax(y)
    #    #print(f'predict is {p}, label is {t}')
    #    #img = (test_X[i] * 256).reshape(28, 28)
    #    #Image.fromarray(np.uint8(img)).show()
    #    counter += 1 if p == t else 0

    # with batch
    for batch_size in [10, 100, 1000, 10000]:
        print(f"batch_size = {batch_size}")
        counter = 0
        t0 = time.time()
        for i in range(0, len(test_X), batch_size):
            x = test_X[i : i + batch_size]
            y = predict(network, x)
            p = np.argmax(y, axis=1)
            # print(f'predict is {p}, label is {test_t[i:i+batch_size]}')
            counter += np.sum(p == test_t[i : i + batch_size])
        t1 = time.time()
        print(f"Accuracy: {float(counter)/len(test_X)}")
        print(f"time: {t1-t0}")
    # print(counter)
