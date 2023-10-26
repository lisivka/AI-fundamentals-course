import numpy as np

class Perceptron:
    def __init__(self, num_inputs, num_hidden_units):
        self.num_inputs = num_inputs
        self.num_hidden_units = num_hidden_units
        self.w_hidden = np.random.rand(num_hidden_units, num_inputs + 1)
        # self.w_out = np.array([-1, 1, -0.5])
        self.w_out = np.random.rand(num_hidden_units + 1)
        print(f"self.w_hidden: {self.w_hidden}")
        print(f"self.w_out: {self.w_out}")
        print(f"num_inputs: {num_inputs}")
        print(f"num_hidden_units: {num_hidden_units}")

    def activation(self, x):
        return 0 if x <= 0 else 1

    def forward(self, x):
        x = np.concatenate((x, [1]))  # Add bias input

        hidden_sum = np.dot(self.w_hidden, x)
        hidden_out = [self.activation(x) for x in hidden_sum]
        hidden_out.append(1)
        hidden_out = np.array(hidden_out)

        output_sum = np.dot(self.w_out, hidden_out)
        y = self.activation(output_sum)
        return y

    def train(self, x, target, learning_rate=0.1):
        x = np.concatenate((x, [1]))  # Add bias input
        hidden_sum = np.dot(self.w_hidden, x)
        hidden_out = [self.activation(x) for x in hidden_sum]
        hidden_out.append(1)
        hidden_out = np.array(hidden_out)

        output_sum = np.dot(self.w_out, hidden_out)
        output = self.activation(output_sum)

        error = target - output

        # Backpropagation
        delta_out = error * (output * (1 - output))
        delta_hidden = delta_out * self.w_out[:-1] * (hidden_out[:-1] * (1 - hidden_out[:-1]))

        self.w_out += learning_rate * delta_out * hidden_out
        self.w_hidden += learning_rate * np.outer(delta_hidden, x)

# Test your Perceptron
C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)]

num_inputs = len(C1[0])
num_hidden_units = 2
perceptron = Perceptron(num_inputs, num_hidden_units)

epochs = 1000

for epoch in range(epochs):
    for data_point in C1:
        perceptron.train(data_point, 1)

    for data_point in C2:
        perceptron.train(data_point, 0)

# Test the trained Perceptron
print(perceptron.forward(C1[0]), perceptron.forward(C1[1]))
print(perceptron.forward(C2[0]), perceptron.forward(C2[1]))
