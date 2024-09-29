# Feedforward Neural Network (FNN) and Feedforward Neural Layer (FNL)

This repository implements a simple Feedforward Neural Network (FNN) and Feedforward Neural Layer (FNL) in Python, using NumPy for matrix operations. The FNN class represents the entire network structure, while the FNL class models each individual layer within the network. This project allows for customization of network architecture, activation functions, normalization functions, and loss functions.

## Features

- **FNN (Feedforward Neural Network)**: 
  - Supports multiple layers with flexible architecture.
  - Forward propagation for computing network outputs.
  - Backward propagation for adjusting weights and biases using gradients.
  
- **FNL (Feedforward Neural Layer)**:
  - Handles the operations of each individual layer, including:
    - Weight and bias initialization.
    - Feedforward computation.
    - Gradient updates during backpropagation.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/repo-name.git
    ```

2. Navigate to the project directory:

    ```bash
    cd repo-name
    ```

3. Install required dependencies:

    ```bash
    pip install numpy
    ```

## Usage

### Example Code

Here’s an example of how to initialize and use the Feedforward Neural Network (FNN):

```python
import numpy as np
from fnn import FNN, FNL

# Example usage
input_size = 4
output_size = 3
hidden_layers = [5, 5]
activation_fn = YourActivationFunction()  # Replace with actual activation function
normalize_fn = YourNormalizationFunction()  # Replace with actual normalization function
loss_fn = YourLossFunction()  # Replace with actual loss function

# Create a neural network
fnn = FNN(input_size, output_size, hidden_layers, activation_fn, normalize_fn, loss_fn)

# Perform forward propagation
input_matrix = np.random.randn(input_size, 10)  # Example input matrix
fnn.forwardPropagate(input_matrix)

# Perform backward propagation
correct_labels = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  # Example correct labels
learning_rate = 0.01
fnn.backwardPropagate(learning_rate, correct_labels)
