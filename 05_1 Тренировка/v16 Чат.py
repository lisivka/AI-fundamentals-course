import numpy as np


class Perceptron:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                input_layer = X[i]
            hidden_layer_input = np.dot(input_layer, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            output_layer_output = self.sigmoid(output_layer_input)

            error = y[i] - output_layer_output
            total_error += error

            d_output = error * self.sigmoid_derivative(output_layer_output)
            error_hidden = d_output * self.weights_hidden_output
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_layer_output)

            self.weights_hidden_output += hidden_layer_output * d_output * learning_rate
            self.weights_input_hidden += np.outer(input_layer, d_hidden) * learning_rate

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Error: {total_error}')

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output


# Пример использования:
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 1, 1, 0])

perceptron = Perceptron(input_size=3, hidden_size=4)
perceptron.train(X, y)

# Проверка обученной сети
print("Результаты после обучения:")
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f'Входные данные: {X[i]}, Прогноз: {prediction}')
print(f"Веса: {perceptron.weights_input_hidden}, {perceptron.weights_hidden_output}")
