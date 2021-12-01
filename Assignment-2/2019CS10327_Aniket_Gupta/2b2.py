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

# get data for a particular digit
def getData(digit, train = True):
    if train:
        n = trainingData.shape[1]
        required = (trainingData[:, n-1]==digit).flatten()
        data = trainingData[required]
        return data/255
    else:
        n = testData.shape[1]
        required = (testData[:, n-1]==digit).flatten()
        data = testData[required]
        return data/255

# function to draw and save images
def drawImage(imgPixels,img_name):
    img = imgPixels.reshape((28, 28))
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.save(img_name)


# get a confusion matrix for given outcomes
def getConfusionMatrix(predicted, actual):
    confusion = np.zeros((10, 10), dtype=np.int32)  #[actual, predicted]
    for i in range(len(predicted)):
        confusion[int(actual[i]), int(predicted[i])]+=1
    return confusion

# function to get misclassified images
def getMissClassifiedImages(conf, predicted, actual):
    image_list = []
    for i in range(10):
        mx = 0
        mis_j = -1
        for j in range(10):
            if i==j:
                continue
            if conf[i,j]>mx:
                mx = conf[i, j]
                mis_j = j
        if mis_j == -1:
            continue
        image_list.append((i,mis_j))
    for i in image_list:
        for j in range(len(predicted)):
            if actual[j] == i[0] and predicted[j]==i[1]:
                drawImage(testData[j,:784], "libsvm_{0}_predictedAs_{1}.png".format(i[0], i[1]))


# main function to handle model training,testing and result analysis
def main(trainFile, testFile, part_c=0):
    global trainingData
    global testData
    trainingData = pd.read_csv(trainFile, header = None).values
    testData = pd.read_csv(testFile, header = None).values
    _, n = trainingData.shape

    trainX = trainingData[:, :(n-1)]/255
    trainY = trainingData[:, (n-1)].reshape((-1,))
    testX = testData[:, :(n-1)]/255
    testY = testData[:, (n-1)].reshape((-1,))
    print()
    print("Using Gaussian Kernel: ")
    startTime = time.time()
    m_gaussian = svm_train(trainY, trainX, '-s 0 -g 0.05 -t 2 -c 1 ')
    endTime = time.time()
    trainingTime = endTime-startTime
    print("Time taken to train multiclass SVM using libsvm library: ", trainingTime)
    #svm_predict(trainY, trainX, m_gaussian)
    testResult = svm_predict(testY, testX, m_gaussian)
    if part_c==1:
        conf = getConfusionMatrix(testResult[0], testY)
        print("Confusion Matrix Obtained:")
        print(conf)
        #getMissClassifiedImages(conf, testResult[0], testY)

# entry point in the code
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile, part_c = arg[0], arg[1], arg[2]
    main(trainFile, testFile, int(part_c))