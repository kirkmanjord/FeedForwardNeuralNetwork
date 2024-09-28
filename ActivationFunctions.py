"""
This module defines abstract and concrete classes for activation functions used in neural networks.

The `ActivationFunction` abstract base class provides the foundation for defining various activation functions
that are commonly used in machine learning and deep learning models. The `ReLu` class implements the ReLu
(Rectified Linear Unit) activation function, which is frequently used due to its effectiveness in handling
non-linearity.

Classes:
    - ActivationFunction: An abstract base class for defining activation functions with methods for
      activation, derivative, and standard deviation calculation for weight initialization.
    - ReLu: A concrete implementation of the ReLu activation function, including methods for calculating
      the activated output, the derivative, and standard deviation for weight initialization.

Dependencies:
    - numpy
    - math

Example Usage:
    relu = ReLu()
    activated_matrix = relu.activate(input_matrix)
    derivative_matrix = relu.derivative(input_matrix)
    std = relu.getSTD(num_inputs, num_outputs)
"""
from abc import ABC, abstractmethod
import numpy as np
import math




class ActivationFunction(ABC):
    """
    An abstract base class for all activation functions.

    This class defines the structure that all activation functions must follow. Activation functions
    are used in neural networks to introduce non-linearity into the model, making it possible to
    learn complex patterns. This class contains abstract methods that need to be implemented by any
    concrete subclass.

    Methods:
        - activate(inputMatrix): Apply the activation function to the input matrix and return the result.
        - derivative(inputMatrix): Compute the derivative of the activation function for backpropagation.
        - getSTD(numOfInputs, numOfOutputs): Return the standard deviation for initializing weights
          based on the number of inputs and outputs, typically for weight initialization strategies.
    """
    @abstractmethod
    def activate(self, inputMatrix):
        """
        Apply the activation function to the input matrix.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the inputs to the activation function.

        Returns:
            numpy.ndarray: A matrix where the activation function has been applied element-wise.
        """
        pass

    @abstractmethod
    def derivative(self,  inputMatrix):
        """
        Compute the derivative of the activation function for backpropagation.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the inputs to the activation function.

        Returns:
            numpy.ndarray: A matrix containing the derivative of the activation function.
        """
        pass

    @abstractmethod
    def getSTD(self,numOfInputs, NumOfOutputs):
        """
        Calculate the standard deviation for weight initialization.

        Args:
            numOfInputs (int): The number of input units.
            NumOfOutputs (int): The number of output units.

        Returns:
            float: The standard deviation for initializing weights.

        """
        pass


class ReLu(ActivationFunction):
    """
    A class used to represent the ReLU (Rectified Linear Unit) activation function.

    The ReLU activation function is widely used in neural networks to introduce
    non-linearity. It outputs the input directly if it's positive, and outputs
    zero for negative inputs. This class also includes methods to compute the
    derivative of the ReLU function, which is used during backpropagation, and
    a method for calculating the standard deviation used for weight initialization.
    """

    def activate(self, inputMatrix):
        """
        Apply the ReLU activation function element-wise to the input matrix.

        The ReLU function outputs the input value if it's positive, and 0 if
        the input is negative. This method applies the ReLU transformation
        across all elements of the input matrix.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the inputs to apply
                                         the ReLU activation function to.

        Returns:
            numpy.ndarray: A matrix where the ReLU function has been applied element-wise.
        """
        matrix = inputMatrix.copy()
        matrix[inputMatrix < 0] = 0
        return matrix

    def derivative(self,  inputMatrix):
        """
        Compute the derivative of the ReLU activation function element-wise.

        The derivative of ReLU is 1 for positive inputs and 0 for non-positive inputs.
        This method applies the ReLU derivative transformation across all elements
        of the input matrix. The derivative is useful for backpropagation during
        neural network training.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the inputs to calculate
                                         the derivative of the ReLU function.

        Returns:
            numpy.ndarray: A matrix where the ReLU derivative has been applied element-wise.
        """
        matrix = inputMatrix.copy()
        matrix[inputMatrix < 0] = 0
        matrix[inputMatrix >= 0] = 1
        return matrix

    def getSTD(self, numOfInputs, NumOfOutputs):
        """
        Calculate the standard deviation for initializing weights using ReLU.

        This method returns the standard deviation for weight initialization
        based on the number of inputs. This is commonly used in neural networks
        to initialize weights when using the ReLU activation function to help
        avoid the vanishing gradient problem.

        Args:
            numOfInputs (int): The number of input units in the layer.
            NumOfOutputs (int): The number of output units in the layer (not used for ReLU initialization).

        Returns:
            float: The standard deviation for initializing weights.
        """
        return math.sqrt(2/numOfInputs)

































