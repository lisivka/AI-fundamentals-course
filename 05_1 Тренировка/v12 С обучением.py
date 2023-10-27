import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size+1
        self.output_size = output_size
        print(f"input_size: {input_size}")
        print(f"hidden_size: {hidden_size}")
        print(f"output_size: {output_size}")


        # Инициализация весов с небольшими случайными значениями
        self.w_hidden = np.random.rand(hidden_size, input_size + 1)
        self.w_out = np.random.rand(output_size, hidden_size + 1)
        print(f"self.w_hidden: {self.w_hidden}")
        print(f"self.w_out: {self.w_out}")

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        x = np.array(list(x) + [1])  # Добавляем bias как последний элемент входа

        hidden_sum = np.dot(self.w_hidden, x)
        hidden_out = self.activation(hidden_sum)
        hidden_out = np.append(hidden_out, 0)  # Добавляем bias как
        # последний элемент выхода слоя

        output_sum = np.dot(self.w_out, hidden_out)
        y = self.activation(output_sum)
        return y

    def train(self, x, target, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            for i in range(len(x)):
                x_i = np.array(list(x[i]) + [1])  # Добавляем bias
                target_i = target[i]

                # Прямое распространение
                hidden_sum = np.dot(self.w_hidden, x_i)
                hidden_out = self.activation(hidden_sum)
                hidden_out = np.append(hidden_out, 1)  # Добавляем bias

                output_sum = np.dot(self.w_out, hidden_out)
                output = self.activation(output_sum)

                # Вычисление ошибки
                error = target_i - output

                # Обратное распространение
                # delta_output = error * self.activation_derivative(output)
                # error_hidden = np.dot(self.w_out.T, delta_output)
                # delta_hidden = error_hidden[:-1] * self.activation_derivative(hidden_out)
                # Обратное распространение
                delta_output = error * self.activation_derivative(output)
                error_hidden = np.dot(self.w_out.T, delta_output)
                delta_hidden = error_hidden[:-1] * self.activation_derivative(hidden_out[:-1])

                # Обновление весов
                self.w_out += learning_rate * np.outer(delta_output, hidden_out)
                self.w_hidden += learning_rate * np.outer(delta_hidden, x_i)

# Пример использования
x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
target = np.array([1, 1, 0, 0])

perceptron = Perceptron(input_size=2, hidden_size=2, output_size=1)
perceptron.train(x, target, learning_rate=0.1, epochs=1000)

# Проверка результатов
for i in range(len(x)):
    result = perceptron.forward(x[i])
    print(f"Input: {x[i]}, Output: {result}")
