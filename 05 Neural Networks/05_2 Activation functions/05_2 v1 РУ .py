import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Сигмоидная активационная функция
    return 1 / (1 + np.exp(-x))

def relu(x):
    # Функция активации ReLU
    return np.maximum(0, x)

def tanh(x):
    # Гиперболический тангенс (tanh)
    return np.tanh(x)

if __name__ == "__main__":
    # Создаем значения x для построения графика
    x = np.linspace(-5, 5, 100)  # Создаем 100 равномерно распределенных значений от -5 до 5

    # Вычисляем значения y для каждой активационной функции
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = tanh(x)

    # Строим графики активационных функций
    plt.figure(figsize=(8, 6))  # Создаем новую фигуру для графика
    plt.plot(x, y_sigmoid, label='Sigmoid', linestyle='-', color='b')
    plt.plot(x, y_relu, label='ReLU', linestyle='--', color='g')
    plt.plot(x, y_tanh, label='tanh', linestyle='-.', color='r')

    # Добавляем метки и заголовок
    plt.xlabel('Входные значения')
    plt.ylabel('Выходные значения')
    plt.title('Графики активационных функций')
    plt.legend()  # Добавляем легенду, чтобы показать названия функций

    # Показываем график
    plt.show()
