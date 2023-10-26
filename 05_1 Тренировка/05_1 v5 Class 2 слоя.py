import numpy as np


class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1,
                 epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for the first and second layers
        self.weights_input_hidden = np.random.rand(self.input_size,
                                                   self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size,
                                                    self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward propagation
            hidden_input = np.dot(X,
                                  self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output,
                                  self.weights_hidden_output) + self.bias_output
            output = self.sigmoid(output_input)

            # Backpropagation
            error_output = y - output
            d_output = error_output * self.sigmoid_derivative(output)

            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_output.T.dot(
                d_output) * self.learning_rate
            self.bias_output += np.sum(d_output, axis=0,
                                       keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0,
                                       keepdims=True) * self.learning_rate

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output,
                              self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        return output


# Пример использования:

# Обучающие данные для XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])

perceptron = TwoLayerPerceptron(input_size=2, hidden_size=4, output_size=1)
perceptron.train(X, y)

# Тестирование обученного перцептрона
print("Testing:")
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f"Input: {X[i]}, Prediction: {prediction[0]}")
