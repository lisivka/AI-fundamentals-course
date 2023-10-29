import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size=2, output_size=1):
        # Инициализация весов случайным образом
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mean_squared_error(self, y, predicted):
        return np.mean((y - predicted) ** 2)

    def train(self, X, y, epochs=1000, learning_rate=0.1, min_mse=0.001):
        for epoch in range(epochs):
            # Пряме розповсюдження
            # Forward propagation
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = self.sigmoid(output_layer_input)

            # Error
            error = y - output_layer_output
            MSE = self.mean_squared_error(y, output_layer_output)
            if MSE < 0.001:
                print(f"Epoch {epoch+1}/{epochs}, Error: {MSE}")
                break


            # Backpropagation
            d_output = error * self.sigmoid_derivative(output_layer_output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_layer_output)


            # Updating weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
            self.bias_output += np.sum(d_output, axis=0) * learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate


    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output


if __name__ == "__main__":
    # XOR dataset: Input features and corresponding labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the perceptron model
    model = Perceptron(input_size=2)
    model.train(X, y, epochs=100000, learning_rate=0.1)

    # Test the trained model
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = model.predict(test_input)
    print("Predictions:")
    for i in range(len(test_input)):
        predicted = predictions[i]
        MSE = model.mean_squared_error(y[i], predicted)
        print(f"Input: {test_input[i]}, Predictions:: {predicted} , MSE: {MSE}")
    print(f"Weight: {model.weights_input_hidden},"
          f"\n {model.weights_hidden_output.T}")
    print(f"bias: {model.bias_hidden},")



