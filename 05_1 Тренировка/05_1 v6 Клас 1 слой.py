import numpy as np


class Perceptron:
    def __init__(self):
        # Инициализируем веса случайными значениями
        self.weights = np.random.rand(3)

    def activate(self, inputs):
        # Суммируем взвешенные входы
        weighted_sum = np.dot(inputs, self.weights)

        # Применяем функцию активации (1 if x >= 0 else 0)
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for i in range(len(X)):
                inputs = np.append(X[i],
                                   1)  # Добавляем 1 для учета смещения (bias)
                target = y[i]
                prediction = self.activate(inputs)

                # Обновляем веса в соответствии с правилом обучения Перцептрона
                error = target - prediction
                self.weights += learning_rate * error * inputs

                # Выводим информацию о процессе обучения
                print(
                    f"Epoch {epoch + 1}, Input: {inputs[:2]}, Target: {target}, Prediction: {prediction}, Weights: {self.weights}")


# Исходные данные
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Пример использования класса Perceptron
if __name__ == "__main__":
    perceptron = Perceptron()
    num_epochs = 10
    learning_rate = 0.1

    perceptron.train(X, y, num_epochs, learning_rate)

