import numpy as np

class Perceptron:
    def __init__(self):
        self.w1 = np.array([-1, 1, 1])
        self.w2 = np.array([1, -1, 1])
        self.w3 = np.array([2, 2, -1])

    def activation(self, x):
        return 1 if x > 0 else 0
        # return 1 / (1 + np.exp(-x))
    def predict(self, X):
        x = np.array(X)
        w_hidden1 = x[0] * self.w1
        w_hidden2 = x[1] * self.w2
        layer_hidden = np.stack(w_hidden1 + w_hidden2)

        layer_hidden_activation = np.array([self.activation(x) for x in layer_hidden])
        layer_out = layer_hidden_activation.dot(self.w3)
        layer_out_activation = self.activation(layer_out)
        print(f"X: {X}")
        print(f"layer_hidden: {layer_hidden}")
        print(f"layer_hidden activation: {layer_hidden_activation}")
        print(f"layer_out: {layer_out}")
        print(f"layer_out_activation: {layer_out_activation}")
        print("*"*20)
        return layer_out_activation

if __name__ == '__main__':
    perceptron = Perceptron()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    print(f"\nExpected [0, 1, 1, 0]: \nGot      {[perceptron.predict(x) for x in X]}")
    print(f"for X =", *X)
