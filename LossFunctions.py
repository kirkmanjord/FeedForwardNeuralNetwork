"""
This module defines abstract and concrete classes for loss functions used in machine learning models,
specifically focusing on the Cross Entropy loss function.

The `LossFunction` abstract base class defines the structure that all loss functions must follow,
requiring the implementation of a derivative method for backpropagation. The `CrossEntropy` class
implements the Cross Entropy loss function, commonly used in classification tasks in neural networks.
It includes a method for calculating the gradient of the loss function with respect to the inputs.

Classes:
    - LossFunction: An abstract base class for defining loss functions with a required derivative method.
    - CrossEntropy: A concrete implementation of the Cross Entropy loss function, including a method for
      calculating the derivative for backpropagation.

Dependencies:
    - numpy
    - math

Example Usage:
    cross_entropy = CrossEntropy()
    gradient = cross_entropy.derivative(predictions, correct_indices)
"""
from abc import ABC, abstractmethod
import numpy as np
import math


class LossFunction(ABC):
    """
    An abstract base class for loss functions in machine learning models.

    This class defines the required structure for any loss function used in model training.
    Loss functions measure the difference between predicted values and actual target values.
    Any subclass of `LossFunction` must implement the `derivative` method to compute the
    gradient of the loss function, which is necessary for backpropagation.

    Methods:
        - derivative(inputMatrix): Compute the derivative of the loss function for backpropagation.
    """

    @abstractmethod
    def derivative(self, inputMatrix):
        """
        Compute the derivative of the loss function for backpropagation.

        Args:
            inputMatrix (numpy.ndarray): A matrix containing the model's predictions.

        Returns:
            numpy.ndarray: A matrix containing the gradient of the loss function.
        """

        pass


class CrossEntropy(LossFunction):
    """
    A concrete implementation of the Cross Entropy loss function.

    The Cross Entropy loss function is commonly used for classification tasks in machine learning,
    particularly for models that output probabilities. It measures the dissimilarity between
    predicted probabilities and the actual class labels.

    Methods:
        - derivative(inputMatrix, listOfCorrectIndices): Computes the gradient of the Cross Entropy
          loss function for a batch of predictions with respect to the input.
    """

    def derivative(self, inputMatrix, listOfCorrectIndices):
        """
        Compute the derivative of the Cross Entropy loss for a batch of predictions.

        This method calculates the gradient of the Cross Entropy loss for each input in
        the batch. It compares the predicted probabilities to the correct class indices,
        computing the derivative of the loss with respect to the input.

        Args:
            inputMatrix (numpy.ndarray): A matrix of shape (num_classes, batch_size) containing
                                         the predicted probabilities for each class.
            listOfCorrectIndices (list): A list of the correct class indices for each sample
                                         in the batch.

        Returns:
            list: A list of numpy arrays representing the derivative of the loss function for
                  each case in the batch.
        """
        batchDerivatives = []
        loss = 0
        correctClass = 0
        maxIndex = 0
        for case in range(inputMatrix.shape[1]):

            col = inputMatrix[:, case: case + 1]
            listVerse = col.T.flatten().tolist()
            max = 0
            maxIndex = 0
            for i in range(len(listVerse)):
                if listVerse[i] > max:
                    max = listVerse[i]
                    maxIndex = i
            #print(maxIndex)
            correctClass = int(listOfCorrectIndices[case])

            matrix = np.zeros((1, col.shape[0]))
            matrix[0, correctClass] = -1 / col[correctClass, 0]
            batchDerivatives.append(matrix)

            if maxIndex == correctClass:
                loss += 1

        loss = loss / inputMatrix.shape[1]
        print(loss)

        # print(loss)

        return batchDerivatives
