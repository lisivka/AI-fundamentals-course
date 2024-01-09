import numpy as np

# Функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации
def sigmoid_derivative(x):
    return x * (1 - x)

# Входные данные
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Выходные данные
outputs = np.array([[0], [1], [1], [0]])

# Инициализация весов
np.random.seed(0)
weights_input_hidden = np.random.random((2, 3)) # 2 входа, 3 нейрона в скрытом слое
weights_hidden_output = np.random.random((3, 1)) # 3 нейрона в скрытом слое, 1 выход
# print(f"Веса скрытого слоя: \n{weights_input_hidden}")
# print(f"Веса выходного слоя: \n{weights_hidden_output}")

# Обучение сети
for _ in range(100000):
    # Прямое распространение
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)
    predicted_output_int = [1 if x > 0.5 else 0 for x in predicted_output]

    # Обратное распространение
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Обновление весов
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output)
    weights_input_hidden += inputs.T.dot(d_hidden_layer)



print(f"Выходные данные после обучения: \n{predicted_output}")
print(f"Выходные данные после обучения: \n{predicted_output_int}")
print(f"Веса скрытого слоя: \n{weights_input_hidden}")
print(f"Веса выходного слоя: \n{weights_hidden_output}")
