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














