import numpy as np

class Perceptron:
    def __init__(self):
        self.w1 = np.array([-1, 1, 1])
        self.w2 = np.array([1, -1, 1])
        self.w3 = np.array([2, 2, -1])

    def activation(self, x):
        return 0 if x <= 0 else 1

    def predict(self, C):
        x = np.array(C)
        w_hidden1 = x[0] * self.w1
        w_hidden2 = x[1] * self.w2
        layer_hidden = np.stack(w_hidden1 + w_hidden2)
        layer_hidden_activation = np.array([self.activation(x) for x in layer_hidden])
        layer_out = layer_hidden_activation.dot(self.w3)
        out = self.activation(layer_out)
        return out

if __name__ == '__main__':
    perceptron = Perceptron()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    print(f"Expected [0, 1, 1, 0]: \nGot      {[perceptron.predict(x) for x in X]}")
