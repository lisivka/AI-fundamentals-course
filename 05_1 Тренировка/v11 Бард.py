import random

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.weights = [random.random() for i in range(2)]
        self.learning_rate = learning_rate

    def train(self, inputs, targets):
        for input, target in zip(inputs, targets):
            output = self.predict(input)
            error = target - output

            for i in range(len(self.weights)):
                self.weights[i] += self.learning_rate * error * input[i]

    def predict(self, input):
        weighted_sum = sum([weight * input[i] for i, weight in enumerate(self.weights)])
        return 1 if weighted_sum > 0 else -1

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [-1, 1, 1, -1]

Perceptron = Perceptron()
Perceptron.train(inputs, targets)

new_input = [1, 0]

prediction = Perceptron.predict(new_input)

print(prediction)
