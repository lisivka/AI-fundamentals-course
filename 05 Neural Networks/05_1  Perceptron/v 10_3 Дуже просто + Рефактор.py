import numpy as np


def activation(x):
    return 0 if x <= 0 else 1


def predict(C):
    x = np.array(C)

    w1 = np.array([-1, 1, 1])
    w2 = np.array([1, -1, 1])
    w3 = np.array([2, 2, -1])

    w_hidden1 = x[0] * w1
    w_hidden2 = x[1] * w2

    layer_hidden = np.stack(w_hidden1 + w_hidden2)
    layer_hidden_activation = np.array([activation(x) for x in layer_hidden])

    layer_out = layer_hidden_activation.dot(w3)
    out = activation(layer_out)

    return out


if __name__ == '__main__':

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    print(f"Expected [0, 1, 1, 0]: \nGot      {[predict(x) for x in X]}")
