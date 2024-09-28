"""
This module defines classes for a simple Feedforward Neural Network (FNN) and Feedforward Neural Layer (FNL).

The `FNN` class represents the full neural network, which consists of multiple layers (`FNL`) and supports
forward and backward propagation. It integrates activation functions, normalization functions, and loss functions
to compute gradients and update weights during training. The `FNL` class defines the structure and operations
for each layer in the network, including weight and bias initialization, feedforward computation, and gradient updates.

Classes:
    - FNN: Represents a feedforward neural network with multiple layers.
    - FNL: Represents a single feedforward layer in the neural network.

Dependencies:
    - numpy

Example Usage:
    fnn = FNN(input_size, output_size, layer_sizes, activation_fn, normalize_fn, loss_fn)
    fnn.forwardPropagate(input_matrix)
    fnn.backwardPropagate(learning_rate, correct_labels)
"""

import numpy as np

class FNN():
    """
    Represents a feedforward neural network with multiple layers.

    The `FNN` class constructs a neural network from a series of layers defined by the `FNL` class. It supports
    forward propagation to compute the output of the network for a given input, and backward propagation for
    updating the weights and biases based on the loss function. The constructor allows customization of the
    network architecture by specifying input size, output size, hidden layers, activation functions,
    normalization functions, and loss functions.

    Methods:
        - appendLayer(layer): Appends a new layer to the network.
        - forwardPropagate(inputMatrix): Performs forward propagation through the network.
        - backwardPropagate(learningRate, listOfCorrectAnswers): Performs backward propagation and updates the weights.
    """
    def __init__(self, inputSize, outputSize, layerSizeList, activationFunction, normalizeFunction, lossFunction):
        """
               Constructor initializes the network layers with the given input size, hidden layer sizes,
               output size, activation function, normalization function, and loss function.

               Args:
                   inputSize (int): Number of input features.
                   outputSize (int): Number of output classes.
                   layerSizeList (list): List of sizes for hidden layers.
                   activationFunction (object): Activation function used between layers.
                   normalizeFunction (object): Normalization function for the final output layer.
                   lossFunction (object): Loss function used during training.
        """
        self.lastLayer = FNL(inputSize,layerSizeList[0],activationFunction)
        self.firstLayer = self.lastLayer
        for i in range(1,len(layerSizeList)):
            inputSize = layerSizeList[i-1]
            self.appendLayer(FNL(inputSize,layerSizeList[i],activationFunction))
        self.appendLayer(FNL(layerSizeList[-1], outputSize, normalizeFunction))
        self.normalizeFunction = normalizeFunction
        self.lossFunction = lossFunction

    def appendLayer(self, layer):
        """
        Appends a new layer to the network.

        This method links a new layer to the last layer in the network and updates the reference
        to the current last layer.

        Args:
            layer (FNL): The new layer to be appended to the network.
        """
        self.lastLayer.setNextLayer(layer)
        layer.setBackLayer(self.lastLayer)
        self.lastLayer = layer


    def forwardPropagate(self, inputMatrix):
        """
        Performs forward propagation through the network.

        This method feeds the input through the first layer and successively passes the output
        to each subsequent layer until the final output is obtained.

        Args:
            inputMatrix (numpy.ndarray): The input matrix to propagate through the network.
        """
        self.firstLayer.feedForward(inputMatrix)


    def backwardPropagate(self, learningRate, listOfCorrectAnswers):
        """
        Performs backward propagation and updates the weights and biases.

        This method computes the gradient of the loss function with respect to the final output,
        combines it with the gradient of the normalization function, and propagates these gradients
        backwards through the network to update the weights and biases of each layer.

        Args:
            learningRate (float): The learning rate used to scale the weight and bias updates.
            listOfCorrectAnswers (list): The correct labels for the current batch of inputs.
        """

        #print(listOfCorrectAnswers)
        lossGradient =  self.lossFunction.derivative(self.lastLayer.getNodeValues(),listOfCorrectAnswers)
        normalizationGradient = self.normalizeFunction.derivative(self.lastLayer.getPreActivation())
        lastLayerGradient = self.normalizeFunction.combineLossDerivate(lossGradient, normalizationGradient)
        self.lastLayer.computeGradient(lastLayerGradient)
        currLayer = self.lastLayer
        while currLayer.hasBackLayer():
            currLayer.updateWeightsAndBiases(learningRate)
            currLayer.flush()
            currLayer = currLayer.backLayer
        currLayer.updateWeightsAndBiases(learningRate)
        currLayer.flush()







class FNL():
    """
    Represents a single feedforward layer in the neural network.

    The `FNL` class defines the structure of each layer in the network. It supports weight and bias initialization,
    feedforward computation for a given input, and gradient updates during backpropagation. The layer is connected to
    previous and next layers, and it performs matrix operations to compute activations and gradients.

    Methods:
        - feedForward(inputMatrix): Performs feedforward computation for this layer.
        - computeGradient(derivativeOfLoss): Computes the gradient for weights and biases based on the loss derivative.
        - updateWeightsAndBiases(learningRate): Updates the weights and biases of the layer using the computed gradients.
        - flush(): Resets internal values for the next forward-backward propagation cycle.
    """

    def __init__(self,numInputs, numNodes, activationFunction):
        """
        Initializes a feedforward neural layer with random weights and biases (according to the activation function).

        Args:
            numInputs (int): Number of input features to the layer.
            numNodes (int): Number of neurons (nodes) in this layer.
            activationFunction (object): The activation function to apply to the layer's output.
        """
        self.numInputs = numInputs
        self.numNodes = numNodes
        self.activationFunction = activationFunction
        self.weights = np.random.randn(numNodes, numInputs) * activationFunction.getSTD(numInputs, numNodes)
        self.biases = np.random.randn(numNodes,1) * activationFunction.getSTD(numInputs, numNodes)
        self.backLayer = None
        self.nextLayer = None
        self.preActivation = None
        self.nodeValues = None
        self.input = None
        self.weightGradient = None
        self.biasGradient = None

    def getPreActivation(self):
        return self.preActivation

    def getNodeValues(self):
        return self.nodeValues
    def hasBackLayer(self):
        return self.backLayer != None


    def feedForward(self, inputMatrix):
        """
        Performs feedforward computation for this layer.

        This method calculates the pre-activation values by applying a dot product between the input
        matrix and the weights, adds the biases, and then applies the activation function to obtain
        the output. The output is passed to the next layer if it exists.

        Args:
            inputMatrix (numpy.ndarray): The input matrix for this layer.

        Returns:
            numpy.ndarray: The output of the layer after applying the activation function.
        """
        self.input = inputMatrix
        self.preActivation = np.dot(self.weights, inputMatrix) + self.biases
        self.nodeValues = self.activationFunction.activate(self.preActivation)
        if self.nextLayer != None:
            return self.nextLayer.feedForward(self.nodeValues)
        else:
            return self.nodeValues

    def setBackLayer(self, backLayer):
        self.backLayer = backLayer

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer

    def flush(self):
        """
        Resets the internal values of the layer for the next forward-backward propagation cycle.

        This method clears the stored node values, input, and gradients in preparation for
        the next forward and backward pass through the network.
        """
        self.nodeValues = None
        self.input = None
        self.weightGradient = None
        self.biasGradient = None
        self.preActivation = None

    def updateWeightsAndBiases(self, learningRate):
        """
        Updates the weights and biases of the layer using the computed gradients.

        This method applies the computed gradients scaled by the learning rate to update
        the weights and biases for this layer.

        Args:
            learningRate (float): The learning rate used to scale the gradient updates.
        """
        self.weights = self.weights - learningRate * self.weightGradient
        self.biases = self.biases - learningRate * self.biasGradient


    def computeGradient(self, derivativeOfLoss):
        """
        Computes the gradient for the weights and biases based on the loss derivative.

        This method calculates the gradient of the loss function with respect to the weights
        and biases of this layer, and propagates the gradient backward to the previous layer.

        Args:
            derivativeOfLoss (numpy.ndarray): The derivative of the loss function with respect to this layer.
        """
        gradient = derivativeOfLoss.T

        self.weightGradient = np.dot(gradient, self.input.T) / gradient.shape[1]
        self.biasGradient = np.sum(gradient.T, axis = 0, keepdims = True).T / gradient.shape[1]
        if self.backLayer != None:
            activatedInputs = self.backLayer.activationFunction.derivative(self.backLayer.getPreActivation().T)
            #activatedInputs =self.backLayer.activationFunction.derivative(self.input.T)
            backLayerDerivative = np.dot(derivativeOfLoss, self.weights)  * activatedInputs
            self.backLayer.computeGradient(backLayerDerivative)




