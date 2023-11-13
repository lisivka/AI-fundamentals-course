import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid activation function.
    Args:
        x (float): Input value.
    Returns:
        float: Output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    Args:
        x (float): Input value.
    Returns:
        float: Output of the ReLU function.
    """
    return np.maximum(0, x)

def tanh(x):
    """
    Hyperbolic tangent (tanh) activation function.
    Args:
        x (float): Input value.
    Returns:
        float: Output of the tanh function.
    """
    return np.tanh(x)

# Create x values for plotting
x = np.arange(-5, 5, 0.1)

# Calculate y values for each activation function
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)


# Plot the activation functions
plt.figure(figsize=(5, 4))
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_tanh, label='Tanh')

# Add labels and title
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Common Activation Functions')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


