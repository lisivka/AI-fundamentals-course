import numpy as np

class Perceptron:
    def __init__(self):
        # self.w1 = np.array([-1, 1, 1])
        # self.w2 = np.array([1, -1, 1])
        # self.w3 = np.array([2, 2, -1])
        self.w1 = np.array([-2, -2, -2])
        self.w2 = np.array([-2, -2, -2])
        self.w3 = np.array([-2, -2, -2])

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

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                prediction = self.predict(x)
                error = target - prediction
                if error != 0:
                    for j in range(3):
                        self.w3[j] += learning_rate * error * x[0]
                        self.w1[j] += learning_rate * error * x[1]
                        # self.w2[j] += learning_rate * error * x[i]
                        print(f"self.w1[{j}] = {self.w1[j]}")
                        print(f"self.w2[{j}] = {self.w2[j]}")
                        print(f"self.w3[{j}] = {self.w3[j]}")


if __name__ == '__main__':
    perceptron = Perceptron()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    num_epochs = 100
    learning_rate = 0.1

    perceptron.train(X, y, num_epochs, learning_rate)
    print(f"Expected [0, 1, 1, 0]: \nGot     {[perceptron.predict(x) for x in X]}")
    print(f"self.w1 = {perceptron.w1}")
    print(f"self.w2 = {perceptron.w2}")
    print(f"self.w3 = {perceptron.w3}")

