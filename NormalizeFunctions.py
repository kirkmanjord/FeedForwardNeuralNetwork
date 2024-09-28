"""
This module defines abstract and concrete classes for normalization functions used in machine learning models.

The `NormalizeFunctions` abstract base class specifies the structure that all normalization functions must follow,
with the requirement to implement both the `normalize` and `derivative` methods. The `SoftMax` class is a concrete
implementation of the SoftMax function, a commonly used normalization function in classification tasks for converting
logits into probabilities. It includes methods for normalization, calculating derivatives, and combining gradients
from loss and normalization.

Classes:
    - NormalizeFunctions: An abstract base class for defining normalization functions.
    - SoftMax: A concrete implementation of the SoftMax function.

Dependencies:
    - numpy
    - math

Example Usage:
    softmax = SoftMax()
    normalized_output = softmax.normalize(input_matrix)
    gradient = softmax.derivative(input_matrix)
"""
from abc import ABC, abstractmethod
import numpy as np
import math

class NormalizeFunctions(ABC):
    """
    An abstract base class for normalization functions in machine learning models.

    This class defines the required structure for any normalization function used in model training.
    It requires subclasses to implement the `normalize` and `derivative` methods. Additionally, it includes
    a method to combine the gradients from the loss function and the normalization function, as well as a method
    to compute the standard deviation for weight initialization.

    Methods:
        - normalize(inputMatrix): Apply the normalization function to the input matrix.
        - derivative(inputMatrix): Compute the derivative of the normalization function.
        - combineLossDerivate(lossGradientList, normalizeGradientList): Combine gradients from the loss and normalization functions.
        - getSTD(numOfInputs, NumOfOutputs): Compute the standard deviation for weight initialization.
    """
    @abstractmethod
    def normalize(self, inputMatrix):
        pass

    @abstractmethod
    def derivative(self, inputMatrix):
        pass
    def combineLossDerivate(self, lossGradientList, normalizeGradientList):
        """
        Combine gradients from the loss function and the normalization function.

        This method multiplies the gradient of the loss function with the gradient of the normalization
        function for each case in the batch to calculate the combined gradient for backpropagation.

        Args:
            lossGradientList (list): A list of matrices containing the gradients from the loss function.
            normalizeGradientList (list): A list of matrices containing the gradients from the normalization function.

        Returns:
            numpy.ndarray: A combined matrix of gradients for backpropagation.
        """
        matrix = np.empty((lossGradientList[0].shape[1], len(lossGradientList)))
        listOfCols = []
        for i in range(len(lossGradientList)):
            temp = lossGradientList[i].dot(normalizeGradientList[i])
            temp = temp
            listOfCols.append(temp)
        return np.concatenate(listOfCols, axis=0)

    def getSTD(self, numOfInputs, NumOfOutputs):
        """
        Compute the standard deviation for weight initialization using the number of inputs.

        This method returns the standard deviation used for initializing weights when using the SoftMax
        normalization function, based on the number of inputs.

        Args:
            numOfInputs (int): The number of input units in the layer.
            NumOfOutputs (int): The number of output units in the layer (not used for this method).

        Returns:
            float: The standard deviation for initializing weights.
        """
        return math.sqrt(2/numOfInputs)


class SoftMax(NormalizeFunctions):
    """
    A concrete implementation of the SoftMax normalization function.

    The SoftMax function converts raw logits into probabilities, commonly used in classification tasks.
    This class implements methods to normalize an input matrix using SoftMax, calculate the derivative of
    the SoftMax function for backpropagation, and perform activation in neural network layers.

    Methods:
        - normalize(inputMatrix): Apply the SoftMax function to the input matrix.
        - activate(inputMatrix): Alias for `normalize`, used for neural network layers.
        - derivative(inputMatrix): Compute the derivative of the SoftMax function.
    """
    def normalize(self, inputMatrix):
        """
        Apply the SoftMax normalization function to the input matrix.

        The SoftMax function converts raw logits into probabilities by exponentiating the inputs
        and then dividing by the sum of the exponentials across each column.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the input logits.

        Returns:
            numpy.ndarray: A matrix where the SoftMax function has been applied element-wise.
        """
        eToTheMatrix = np.exp(inputMatrix)
        columnSums = np.sum(eToTheMatrix, axis=0, keepdims=True)
        return eToTheMatrix / columnSums
    def activate(self, inputMatrix):
        """
                Apply the SoftMax normalization function to the input matrix.

                The SoftMax function converts raw logits into probabilities by exponentiating the inputs
                and then dividing by the sum of the exponentials across each column.

                Args:
                    inputMatrix (numpy.ndarray): A matrix containing the input logits.

                Returns:
                    numpy.ndarray: A matrix where the SoftMax function has been applied element-wise.
                """

        eToTheMatrix = np.exp(inputMatrix)
        columnSums = np.sum(eToTheMatrix, axis=0, keepdims=True)
        return eToTheMatrix / columnSums
    def derivative(self, inputMatrix):
        """
        Compute the derivative of the SoftMax function for backpropagation.

        This method calculates the Jacobian matrix of the SoftMax function for each case in the batch,
        used for backpropagation in neural networks. The derivative helps adjust the model's weights during training.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the input logits.

        Returns:
            list: A list of numpy arrays representing the Jacobian matrix for each case in the batch.
        """

        softMaxed = self.normalize(inputMatrix)
        batchDerivatives = []

        for case in range(inputMatrix.shape[1]):
            col = softMaxed[:, case : case + 1]
            outputMatrix = np.empty((col.shape[0], col.shape[0]))
            for i in range(col.shape[0]):
                for k in range(col.shape[0]):
                    if (i == k):
                        outputMatrix[i,k] = col[k] - col[i] ** 2
                    else:
                        outputMatrix[i,k] = - col[i] * col[k]
            batchDerivatives.append(outputMatrix)
        return batchDerivatives




