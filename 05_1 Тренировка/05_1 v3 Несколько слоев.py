import numpy as np

# Задаем входные данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Ожидаемые выходные данные XOR
Y = np.array([0, 1, 1, 0])

# Задаем веса и смещения для скрытого слоя и выходного слоя
hidden_layer = np.array([[1, 1], [1, 1]])
output_layer = np.array([1, -2])
print(f"hidden_layer: {hidden_layer}")
print(f"output_layer: {output_layer}")


# Функция активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Прямое распространение
def forward_propagation(x):
    hidden_input = np.dot(x, hidden_layer)
    hidden_output = sigmoid(hidden_input)
    output = sigmoid(np.dot(hidden_output, output_layer))
    return hidden_output, output

# Функция потерь (среднеквадратичная ошибка)
def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Обучение перцептрона
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    for x, y in zip(X, Y):
        # Прямое распространение
        output = forward_propagation(x)

        # Обратное распространение (обновление весов)
        hidden_output, output = forward_propagation(
            x)  # Добавляем hidden_output возвращаемое значение

        # Обратное распространение (обновление весов)
        # Обратное распространение (обновление весов)
        output_error = y - output
        output_delta = output_error * output * (1 - output)
        hidden_layer_error = output_delta @ output_layer  # Используем @ вместо dot
        hidden_layer_delta = hidden_layer_error * hidden_output * (
                    1 - hidden_output)

        output_layer += hidden_output.reshape(-1,
                                              1) * output_delta * learning_rate

        # Перепишем обновление весов для скрытого слоя без использования dot
        for i in range(hidden_layer.shape[0]):
            for j in range(hidden_layer.shape[1]):
                hidden_layer[i, j] += x[i] * hidden_layer_delta[
                    j] * learning_rate

    if (epoch + 1) % 1000 == 0:
        loss = mean_squared_error(Y, forward_propagation(X))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Проверка обученной сети
for x in X:
    prediction = forward_propagation(x)
    print(f"Input: {x}, Output: {prediction}")
