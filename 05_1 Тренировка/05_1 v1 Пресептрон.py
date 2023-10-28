import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.05, epochs=5000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def activation(self, x):
        # Простая ступенчатая функция активации (пороговая функция)
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Вычисление выхода перцептрона
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                prediction = self.predict(x)
                error = target - prediction
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

    def evaluate(self, X, y):
        correct = 0
        for i in range(len(X)):
            x = X[i]
            prediction = self.predict(x)
            if prediction == y[i]:
                correct += 1
        accuracy = correct / len(X)
        return accuracy

# Создание набора данных XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Создание и обучение перцептрона
perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

# Оценка производительности перцептрона
accuracy = perceptron.evaluate(X, y)
print("Accuracy:", accuracy)

# Проверка прогнозов перцептрона
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction}")

print(f"weights: {perceptron.weights}")
print(f"bias: {perceptron.bias}")
