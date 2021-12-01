from numpy import linalg
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys
from libsvm.svmutil import *
import time
from PIL import Image

trainingData = {}
testData = {}
trainX = {}
trainY = {}
testX = {}
testY = {}

# function to get cross validation result for a given value of C
def crossValidate(c, k=5):
    m,n = trainX.shape
    batchSize = int(m/k)
    cvScore = 0
    print()
    print("Validating for C = {0}:".format(c))
    for i in range(k):
        validationSet = trainY[i*batchSize:min((i+1)*batchSize, m)], trainX[i*batchSize:min((i+1)*batchSize, m), :] 
        if i == 0:
            trainingSet = trainY[(i+1)*batchSize:], trainX[(i+1)*batchSize:, :] 
        elif i == k-1:
            trainingSet = trainY[:i*batchSize], trainX[:i*batchSize, :]
        else:
            trainingSet = np.concatenate((trainY[:i*batchSize], trainY[(i+1)*batchSize:])) , np.vstack((trainX[:i*batchSize, :],trainX[(i+1)*batchSize:, :]))
        
        m_gaussian = svm_train(trainingSet[0], trainingSet[1], '-t 2 -c {0} -g 0.05 -q'.format(c))
        testResult = svm_predict(validationSet[0], validationSet[1], m_gaussian, '-q')
        print("**Accuracy on {0} validation set: {1}%".format(i+1, testResult[1][0]))
        cvScore += testResult[1][0]
    cvScore/=k
    print(">> Average accuracy on validation sets: {0}%".format(cvScore))
    return cvScore


#main function to get perform 5-fold valudation and finding test accuracy
def main(trainFile, testFile):
    global trainingData, testData, trainX, trainY, testX, testY
    trainingData = pd.read_csv(trainFile, header = None).values
    testData = pd.read_csv(testFile, header = None).values
    m, n = trainingData.shape
    idx = np.arange(m)
    np.random.shuffle(idx)
    trainingData = trainingData[idx, :]
    
    trainX = trainingData[:, :(n-1)]/255
    trainY = trainingData[:, (n-1)].reshape((-1,))

    testX = testData[:, :(n-1)]/255
    testY = testData[:, (n-1)].reshape((-1,))
    c_list = [1e-5, 1e-3, 1, 5, 10]

    cvScore = []
    accuracy = []
    for c in c_list:
        kFoldCVscore = crossValidate(c, k = 5)
        model = svm_train(trainY, trainX, '-t 2 -c {0} -g 0.05 -q'.format(c))
        testResult = svm_predict(testY, testX, model, '-q')
        print(">> Accuracy on test data for model trained with c = {0} : {1}%".format(c, testResult[1][0]))
        cvScore.append(kFoldCVscore)
        accuracy.append(testResult[1][0])
    print(cvScore)
    print(accuracy)

# entry point in the code
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile = arg[0], arg[1]
    main(trainFile, testFile)