import numpy as np
import FNL
import ActivationFunctions as AF
import NormalizeFunctions as NF
import LossFunctions as LF

dataSet = np.loadtxt(r'C:\Users\rooki\PycharmProjects\pythonProject1\trainer.csv', delimiter=',')
inputSize = dataSet.shape[1] -1
outputSize = 10
layerSizeList = [300,300,300]
activationFunction = AF.ReLu()
normalizeFunction = NF.SoftMax()
lossFunction = LF.CrossEntropy()

batch = dataSet[:10000,1:].T/100
answer = dataSet[:10000,0].tolist()
network = FNL.FNN(inputSize, outputSize, layerSizeList, activationFunction, normalizeFunction, lossFunction)



while True:
    i = 0
    while i < 10000:
        i += 100



        batch = dataSet[i-100:i, 1:].T / 100
        answer = dataSet[i-100:i, 0].tolist()


        network.forwardPropagate(batch)
        network.backwardPropagate(0.01,answer)













