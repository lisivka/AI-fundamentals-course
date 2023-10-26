class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0] * (input_size + 1)  # +1 for the bias
        self.history = {'loss': []}

    def predict(self, inputs):
        # Compute the weighted sum of inputs
        weighted_sum = self.weights[0]  # Initialize with bias
        for i in range(self.input_size):
            weighted_sum += inputs[i] * self.weights[i + 1]

        # Apply the step function as the activation function
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def train(self, training_data):
        for epoch in range(self.epochs):
            total_error = 0
            for data in training_data:
                inputs = data[:-1]
                target = data[-1]
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += error

                # Update the weights
                self.weights[0] += self.learning_rate * error  # Update bias
                for i in range(self.input_size):
                    self.weights[i + 1] += self.learning_rate * error * inputs[i]

            # Calculate mean squared error for the epoch
            mean_squared_error = total_error ** 2 / len(training_data)
            self.history['loss'].append(mean_squared_error)

    def print_weights(self):
        print("Learned weights:")
        for i, weight in enumerate(self.weights):
            print(f"Weight {i}: {weight}")

# Пример использования:

# Обучающие данные для XOR (первые два элемента - входы, третий - выход)
training_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

perceptron = Perceptron(input_size=2)
perceptron.train(training_data)

# Вывод весов после обучения
perceptron.print_weights()

# Тестирование обученного перцептрона
print("Testing:")
for data in training_data:
    inputs = data[:-1]
    target = data[-1]
    prediction = perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Target: {target}, Prediction: {prediction}")
