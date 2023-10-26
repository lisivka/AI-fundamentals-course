import numpy as np

class Perceptron:
    def __init__(self, input_size):
        """
        Initializes a simple perceptron model.

        Args:
            input_size (int): Number of input features.
        """
        self.weights = np.zeros(input_size + 1)




    def activation(self, x):
        """
        Activation function (Step function).

        Args:
            x (float): Input value.

        Returns:
            int: 1 if x >= 0, else 0.
        """
        return result

    def predict(self, x):
        """
        Predicts the output label using the perceptron model.

        Args:
            x (ndarray): Input features.

        Returns:
            int: Predicted label (1 or 0).
        """

        return result

    def train(self, X, y, num_epochs, learning_rate):
        """
        Trains the perceptron model on the given dataset using the perceptron learning rule.

        Args:
            X (ndarray): Input features of the dataset.
            y (ndarray): Ground truth labels of the dataset.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight update.
        """


        return None

# XOR dataset: Input features and corresponding labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create and train the perceptron model
perceptron = Perceptron(input_size=3)
perceptron.train(X, y, num_epochs=1000, learning_rate=0.1)

# Test the trained model
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    prediction = perceptron.predict(data)
    print(f"Input: {data}, Prediction: {prediction}")
