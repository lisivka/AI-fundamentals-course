import numpy as np

class Perceptron:
    def __init__(self):
        # self.w1 = np.array([-1, 1, 1])
        # self.w2 = np.array([1, -1, 1])
        # self.w3 = np.array([2, 2, -1])

        self.w1 = np.random.rand(3)
        self.w2 = np.random.rand(3)
        self.w3 = np.random.rand(3)
        print(f"STARTw3: {self.w3}")

    def activation(self, x):
        return 1 / (1 + np.exp(-x))
        # return 1 if x > 0 else 0
    def predict(self, X):
        x = np.array(X)
        w_hidden1 = x[0] * self.w1
        w_hidden2 = x[1] * self.w2
        layer_hidden = np.stack(w_hidden1 + w_hidden2)

        layer_hidden_activation = np.array([self.activation(x) for x in layer_hidden])
        layer_out = layer_hidden_activation.dot(self.w3)
        layer_out_activation = self.activation(layer_out)
        return layer_out_activation
    def sigmoid_dx(self, x):

        return x * (1 - x)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                input_data = X[i]
                target = y[i]

                # Прямой проход, аналогично методу predict
                x = np.array(input_data)
                w_hidden1 = x[0] * self.w1
                w_hidden2 = x[1] * self.w2
                layer_hidden = np.stack(w_hidden1 + w_hidden2)

                layer_hidden_activation = np.array([self.activation(x) for x in layer_hidden])
                layer_out = layer_hidden_activation.dot(self.w3)
                train_out = self.activation(layer_out)

                # train_out = self.predict(input_data)

                # Рассчитать ошибку
                error = target - train_out


                # Обратный проход (обновление весов)
                delta = error * self.sigmoid_dx(train_out)
                # print(f"\nX: {input_data}")
                # print(f"out: {train_out}")
                # print(f"error: {error}")
                # print(f"sigmoid_dx: {self.sigmoid_dx(train_out)}" )
                # print(f"delta: {delta}")


                # Обновление весов w3
                # print(f"ДОself.w3: {self.w3}")
                self.w3 += learning_rate * delta * train_out
                # print(f"ПОСЛЕself.w3: {self.w3[0]}")
                self.w3[0] -=layer_hidden_activation[0] * learning_rate * delta
                self.w3[1] -=layer_hidden_activation[1] * learning_rate * delta
                self.w3[2] -=layer_hidden_activation[2] * learning_rate * delta

                self.w1[0] -= input_data[0] * learning_rate * delta
                self.w1[1] -= input_data[0] * learning_rate * delta
                self.w1[2] -= input_data[0] * learning_rate * delta

                self.w2[0] -= input_data[1] * learning_rate * delta
                self.w2[1] -= input_data[1] * learning_rate * delta
                self.w2[2] -= input_data[1] * learning_rate * delta



                # Обратный проход для w1 и w2
                # delta_hidden = delta * self.w3
                # delta_hidden1 = delta_hidden[0:2] * layer_hidden_activation[0] * (1 - layer_hidden_activation[0])
                # delta_hidden2 = delta_hidden[2:4] * layer_hidden_activation[1] * (1 - layer_hidden_activation[1])
                #
                # self.w1 += learning_rate * x[0] * delta_hidden1
                # self.w2 += learning_rate * x[1] * delta_hidden2

if __name__ == '__main__':
    perceptron = Perceptron()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    perceptron.train(X, y, learning_rate=0.2, epochs=100000)

    result = [perceptron.predict(x) for x in X]
    print("\n","*"*20)
    print(f"\nExpected [0, 1, 1, 0]: \nGot      {result}")
    print(f"for X =", *X)

    print("\nTrained weights:")
    print("w1:", perceptron.w1)
    print("w2:", perceptron.w2)
    print("w3:", perceptron.w3)