import numpy as np
import sys

def activation_funk(x):
    return 1 / (1 + np.exp(-x))

mapper = np.vectorize(activation_funk)

def MSE(y, Y):
    return np.mean((y - Y) ** 2)

def predict(inputs):
    inputs_1 = np.dot(weights_0_1, inputs)
    outputs_1 = mapper(inputs_1)

    inputs_2 = np.dot(weights_1_2, outputs_1)
    outputs_2 = mapper(inputs_2)

    return outputs_2

def train(inputs, expected_predict):
    inputs_1 = np.dot(weights_0_1, inputs)
    outputs_1 = mapper(inputs_1)

    inputs_2 = np.dot(weights_1_2, outputs_1)
    outputs_2 = mapper(inputs_2)

    actual_predict = outputs_2[0]

    error_layer_2 = np.array([actual_predict - expected_predict])
    gradient_layer_2 = actual_predict * (1 - actual_predict)
    weights_delta_layer_2 = error_layer_2 * gradient_layer_2
    calc_weights_1_2 = weights_1_2 - (
            np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1))) * learning_rate)

    error_layer_1 = weights_delta_layer_2 * weights_1_2
    gradient_layer_1 = outputs_1 * (1 - outputs_1)
    weights_delta_layer_1 = error_layer_1 * gradient_layer_1
    calc_weights_0_1 = weights_0_1 - (np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * learning_rate)

    return calc_weights_0_1, calc_weights_1_2

data = [
    ([0, 0], 0),
    ([0, 0], 0),
    ([0, 1], 0),
    ([0, 1], 0),
    ([1, 0], 1),
    ([1, 0], 1),
    ([1, 1], 0),
    ([1, 1], 0),
]

weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (3, 2))
weights_1_2 = np.random.normal(0.0, 1, (1, 2))

epochs = 3000
learning_rate = 0.07

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stats, correct_predict in data:
        weights_0_1, weights_1_2 = train(np.array(input_stats), correct_predict)
        inputs_.append(np.array(input_stats))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(predict(np.array(inputs_).T), np.array(correct_predictions))

    progress = str(100 * e / float(epochs))[:4]
    train_loss = str(train_loss)[:5]
    sys.stdout.write("\rProgress: " + progress + " Training loss: " + train_loss),

print("\nTraining completed!")

# Prediction and results output

for input_stats, correct_predict in data:
    input = str(input_stats)
    prediction = str(predict(np.array(input_stats)) > 0.5)
    correct_predict = str(correct_predict == 1)

    print("For input: " + input + " the prediction is: " + prediction + ", expected: " + correct_predict)

for input_stats, correct_predict in data:
    input = str(input_stats)
    prediction = str(predict(np.array(input_stats)))
    correct_predict = str(correct_predict == 1)

    print("For input: " + input + " the prediction is: " + prediction + ", expected: " + correct_predict)

print("Final weights_0_1:")
print(weights_0_1)
print("Final weights_1_2:")
print(weights_1_2)
