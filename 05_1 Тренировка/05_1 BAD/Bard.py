import numpy as np

class Perceptron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def predict(self, inputs):
        """
        Predicts the output of the perceptron for the given inputs.

        Args:
            inputs: A NumPy array of input signals.

        Returns:
            The output of the perceptron, which is 1 if the weighted sum of the inputs exceeds the threshold, and 0 otherwise.
        """

        output = np.dot(self.weights, inputs)
        if output >= 0:
            return 1
        else:
            return 0



if __name__ == '__main__':
    # # Create a perceptron with two input signals and three neurons
    # perceptron = Perceptron([0.5, 0.5, 0.5], 0.5)
    #
    # # Predict the output for the input signals [1, 1]
    # output = perceptron.predict([1, 1])
    #
    # # Print the output
    # print(output)



    # Create a perceptron with two input signals and three neurons
    perceptron = Perceptron([0.5, 0.5, 0.5], 0.5)

    # Create NumPy arrays of the input signals and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Make predictions for the input signals
    predictions = perceptron.predict(X)

    # Print the accuracy
    accuracy = np.sum(predictions == y) / len(predictions)
    print(accuracy)




