import numpy as np


class Perceptron:
    def __init__(self, input_size, hidden_size):
        # Инициализация весов случайными значениями
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_layer_weights = np.random.rand(input_size, hidden_size)
        self.hidden_layer_weights = np.random.rand(hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                # Прямое распространение
                input_layer_output = X[i]
                hidden_layer_input = np.dot(input_layer_output, self.input_layer_weights)
                hidden_layer_output = self.sigmoid(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.hidden_layer_weights)
                output_layer_output = self.sigmoid(output_layer_input)

                # Вычисление ошибки
                error = y[i] - output_layer_output

                # Обратное распространение
                d_output = error * self.sigmoid_derivative(output_layer_output)
                error_hidden = np.array([d_output]).dot(self.hidden_layer_weights.T)
                d_hidden = error_hidden * self.sigmoid_derivative(hidden_layer_output)

                # Обновление весов
                self.hidden_layer_weights += hidden_layer_output * d_output * learning_rate
                self.input_layer_weights += np.outer(input_layer_output, d_hidden) * learning_rate

    def predict(self, X):
        # Прямое распространение для предсказания
        hidden_layer_input = np.dot(X, self.input_layer_weights)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.hidden_layer_weights)
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output


# Пример использования:
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    perceptron = Perceptron(2, 4)  # Создаем перцептрон с 2 входами и 4 скрытыми нейронами
    perceptron.train(X, y, learning_rate=0.1, epochs=10000)  # Обучаем перцептрон
    for i in range(len(X)):
        output = perceptron.predict(X[i])  # Предсказываем для каждого входа
        print(f"Input: {X[i]}, Predicted Output: {output}")
