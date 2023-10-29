import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов случайным образом
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mean_squared_error(self, y, predicted):
        return np.mean((y - predicted) ** 2)

    def mean_squared_error_derivative(self, y, predicted):
        return -2 * (y - predicted)

    def train(self, X, y, epochs=100000, learning_rate=0.1):
        for epoch in range(epochs):
            # Прямое распространение
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = self.sigmoid(output_layer_input)

            # Ошибка и её производная
            error = self.mean_squared_error(y, output_layer_output)
            d_error = self.mean_squared_error_derivative(y, output_layer_output)

            # Backpropagation
            d_output = d_error * self.sigmoid_derivative(output_layer_output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_layer_output)

            # Обновление весов и смещений
            self.weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
            self.bias_output += np.sum(d_output, axis=0) * learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output

# Пример использования
if __name__ == "__main__":
    # Пример обучающих данных
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Создаем и тренируем перцептрон
    input_size = 2
    hidden_size = 4
    output_size = 1
    model = Perceptron(input_size, hidden_size, output_size)
    model.train(X, y)

    # Пример предсказания
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = model.predict(test_input)
    print("Предсказания:")
    for i in range(len(test_input)):
        print(f"Вход: {test_input[i]}, Предсказание: {predictions[i]}")
