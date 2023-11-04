import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf

def create_neural_network(input_dim, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression tasks
    model.compile(optimizer='adam', loss='mean_squared_error')  # You can change the optimizer and loss function as needed
    return model

def train_neural_network(model, X_train, y_train, num_epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)
    return history

# Example data and labels
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Hyperparameters to experiment with
input_dim = X_train.shape[1]
learning_rates = [0.001, 0.01, 0.1]
hidden_units_list = [4, 8, 16]
num_epochs = 100
batch_size = 2

# Perform hyperparameter tuning
for learning_rate in learning_rates:
    for hidden_units in hidden_units_list:
        model = create_neural_network(input_dim=input_dim, hidden_units=hidden_units)
        train_neural_network(model, X_train, y_train, num_epochs=num_epochs, batch_size=batch_size)

        # Evaluate the model's performance (you can replace this with your own evaluation metric)
        loss = model.evaluate(X_train, y_train)
        print("\n", "*" * 50)
        print(f"Learning Rate: {learning_rate}, Hidden Units: {hidden_units}, Loss: {loss}")
        for i in X_train:
            predictions = model.predict(np.array([i]))
            print(f"{i} predictions : {predictions}")
