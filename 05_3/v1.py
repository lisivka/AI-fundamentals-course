import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def create_neural_network(input_dim, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression tasks
    model.compile(optimizer='adam', loss='mean_squared_error')  # You can change the optimizer and loss function as needed
    return model

def train_neural_network(model, X_train, y_train, num_epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)
    return history

# Example data and labels
X_train = np.random.rand(100, 10)  # Replace with your actual training data
y_train = np.random.rand(100)  # Replace with your actual training labels

# Hyperparameters to experiment with
input_dim = X_train.shape[1]
hidden_units = 32
num_epochs = 50
batch_size = 32

# # Example data and labels
# X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_train = np.array([0, 1, 1, 0])
#
# # Hyperparameters to experiment with
# learning_rates = [0.001, 0.01, 0.1]
# hidden_units = [4, 8, 16]
# num_epochs = 50
# batch_size = 2

# Create and compile the neural network model
model = create_neural_network(input_dim, hidden_units)

# Train the model
history = train_neural_network(model, X_train, y_train, num_epochs, batch_size)

# You can access training history and evaluate the model as needed
