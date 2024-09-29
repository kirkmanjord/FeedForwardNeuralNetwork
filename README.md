# Feedforward Neural Network (FNN) and Feedforward Neural Layer (FNL)

This repository implements a Feedforward Neural Network (FNN) and Feedforward Neural Layer (FNL) in Python, using NumPy for matrix operations. The FNN class represents the entire network structure, while the FNL class models each individual layer within the network. This project allows for customization of network architecture, activation functions, normalization functions, and loss functions.

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
import FNL
import ActivationFunctions as AF
import NormalizeFunctions as NF
import LossFunctions as LF
#Load data from file
dataSet = np.loadtxt(r'C:\Users\rooki\PycharmProjects\pythonProject1\trainer.csv', delimiter=',')
#determine the amount of inputs
inputSize = dataSet.shape[1] -1
#determine your output size
outputSize = 10
#determine your structure of your Network (each row signifies the layers and the numbers signify nodes)
layerSizeList = [300,300,300]
#Specifying mathematical modules
activationFunction = AF.ReLu()
normalizeFunction = NF.SoftMax()
lossFunction = LF.CrossEntropy()


#Instantiate the FNN
network = FNL.FNN(inputSize, outputSize, layerSizeList, activationFunction, normalizeFunction, lossFunction)

#iterate through batchs
while True:
    i = 0
    while i < 10000:
        i += 100



        batch = dataSet[i-100:i, 1:].T / 100
        answer = dataSet[i-100:i, 0].tolist()

        #feed the batch forward
        network.forwardPropagate(batch)
        #back propogate by identifying learning rate and answers
        network.backwardPropagate(0.01,answer)














