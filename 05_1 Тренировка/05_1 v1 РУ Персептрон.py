# https://www.youtube.com/watch?v=gDvDwH4dJFI

import numpy as np
class Perceptron:
    def __init__(self, input_size):
        # Инициализация модели персептрона
        # input_size - количество входных признаков
        self.weights = np.empty(input_size + 1)  # Создаем веса для каждого
        # входа и добавляем вес смещения (bias)

    def activation(self, x):
        # Функция активации (функция порога)
        return 1 if x >= 0 else 0
        # return 1 / (1 + np.exp(-x))
    def predict(self, x):
        # Прогнозирование метки класса с использованием модели персептрона
        # x - входные признаки
        # Добавляем 1 в начало вектора x для учета веса смещения (bias)
        x_with_bias = np.insert(x, 0, 1)
        z = np.dot(self.weights, x_with_bias)  # Вычисляем взвешенную сумму
        prediction = self.activation(z)  # Применяем функцию активации
        return prediction

    def train(self, X, y, num_epochs, learning_rate):
        # Обучение модели персептрона на наборе данных
        # X - входные признаки
        # y - истинные метки классов
        # num_epochs - количество эпох обучения
        # learning_rate - скорость обучения

        for epoch in range(num_epochs):
            for i in range(len(X)):
                # Для каждого примера из обучающего набора
                x = X[i]
                target = y[i]
                # Добавляем 1 в начало вектора x для учета веса смещения (bias)
                x_with_bias = np.insert(x, 0, 1)
                print(x_with_bias)
                z = np.dot(self.weights, x_with_bias)
                prediction = self.activation(z)

                # Вычисляем разницу между предсказанным и истинным значением
                error = target - prediction

                # Обновляем веса согласно правилу обучения персептрона
                self.weights += learning_rate * error * x_with_bias

# XOR датасет: Входные признаки и соответствующие метки
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Создаем и обучаем модель персептрона
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, num_epochs=1000, learning_rate=0.1)

# Проверяем обученную модель
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    prediction = perceptron.predict(data)
    print(f"Входные признаки: {data}, Прогноз: {prediction}")
