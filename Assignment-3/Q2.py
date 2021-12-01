from numpy import linalg
from numpy.core.arrayprint import _leading_trailing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.neural_network import MLPClassifier

np.random.seed(1)

def hot_encoded_poker_data(data):
    hot_encoded_dataX = np.zeros((data.shape[0], 85))
    hot_encoded_dataY = np.zeros((data.shape[0], 10))
    dataX, dataY = data[:, :-1], data[:, -1]
    m,n = dataX.shape
    for i in range(m):
        count = 0
        for j in range(n):
            if j%2==0:
                hot_encoded_dataX[i, count + data[i, j] - 1] = 1
                count += 4
            else:
                hot_encoded_dataX[i, count + data[i, j] - 1] = 1
                count += 13
        hot_encoded_dataY[i, dataY[i]] = 1
    return hot_encoded_dataX, hot_encoded_dataY

def sigmoid(data):
    return 1/(1+np.exp(-data))

def d_sigmoid(data):
    return data*(1-data)

def relu(data):
    return np.maximum(0, data)

def d_relu(data):
    return np.where(data>0, 1, 0)

class neuralNetwork():
    def __init__(self, hidden_layer_arch, M, n, r, activation_fn = "sigmoid"):
        self.layerUnits = [n] + hidden_layer_arch + [r]
        self.theta = [None for i in range(len(self.layerUnits))]
        self.activationUnit = activation_fn
        self.M = M
        self.n_features = n
        self.n_output = r

    def getLearningRate(self, epoch, learningRate, adaptive):
        if adaptive:
            return 0.1/epoch**0.5
        else:
            return learningRate


    def train(self, TrainData_X, trainData_Y, learningRate = 0.1, adaptive = False):
        m, n = TrainData_X.shape
        trainDataX = np.hstack((np.ones((m,1)), TrainData_X))
        converged = False
        theta = [None] + [np.random.uniform(-1, 1, (self.layerUnits[i]+1, self.layerUnits[i+1]))*(2/self.layerUnits[i])**0.5 for i in range(len(self.layerUnits) - 1)]

        layer_output = [None for i in range(len(self.layerUnits))]
        delta = [None for i in range(len(self.layerUnits))]
        derivative  = [None for i in range(len(self.layerUnits))]
        n_layers = len(self.layerUnits)
        M = self.M
        last_error = 0
        threshold = 1e-8
        iter_arr = []
        error_arr = []
        iterr = 0
        epoch = 1

        
        
        while(not converged):
            curr_epoch_err = 0
            random_order = np.random.permutation(m)
            trainDataX = trainDataX[random_order]
            trainData_Y = trainData_Y[random_order]
            for start in range(0, m, M):
                end = min(start+M, m)
                batchSize = end - start
                
                batchTrainData_X = trainDataX[start:end, :]
                batchTrainData_Y = trainData_Y[start:end, :]
                iterr += 1

                layer_output[0] = batchTrainData_X
                for i in range(1, n_layers):
                    if i == n_layers-1:
                        layer_output[i] = np.hstack((np.ones((batchSize,1)),sigmoid(layer_output[i-1].dot(theta[i]))))
                        continue
                    if self.activationUnit == 'sigmoid':
                        layer_output[i] = np.hstack((np.ones((batchSize,1)),sigmoid(layer_output[i-1].dot(theta[i]))))
                    else:
                        layer_output[i] = np.hstack((np.ones((batchSize,1)),relu(layer_output[i-1].dot(theta[i]))))

                
                delta[-1] = (batchTrainData_Y - layer_output[-1][:,1:]).T * d_sigmoid(layer_output[-1][:, 1:].T)/batchSize
                for i in range(n_layers-2, 0, -1):
                    if self.activationUnit == 'sigmoid':
                        delta[i] = (theta[i+1][1:, :].dot(delta[i+1])) * d_sigmoid(layer_output[i][:, 1:].T)
                    else:
                        delta[i] = (theta[i+1][1:, :].dot(delta[i+1])) * d_relu(layer_output[i][:, 1:].T)


                for i in range(1, len(self.layerUnits)):
                    derivative[i] = -(delta[i] @ layer_output[i-1]).T

                for i in range(1, n_layers):
                    theta[i] -= self.getLearningRate(epoch, learningRate, adaptive) * derivative[i]
                
                new_error = np.sum((layer_output[-1][:, 1:]-batchTrainData_Y)**2)/(2 * batchSize)
                curr_epoch_err += new_error
                iter_arr.append(iterr)
                error_arr.append(new_error)
            
            curr_epoch_err/=(m//M)
            if (abs(curr_epoch_err-last_error) < threshold or epoch>1000):
                converged = True
            epoch += 1
            last_error = curr_epoch_err

        self.theta = theta
        # print(len(error_arr), len(iter_arr))
        plt.plot(iter_arr, error_arr)
        plt.show()
        return

    def predict(self, testData_X):
        m = testData_X.shape[0]
        layer_input = np.hstack((np.ones((testData_X.shape[0],1)), testData_X))
        n_layers = len(self.layerUnits)
        for i in range(1, n_layers):
            if self.activationUnit == 'sigmoid' or i == n_layers-1:
                layer_input = np.hstack((np.ones((m,1)),sigmoid((layer_input).dot(self.theta[i]))))
            else:
                layer_input = np.hstack((np.ones((m,1)),relu((layer_input).dot(self.theta[i]))))
        return np.argmax(layer_input[:, 1:], axis = 1)
    

    def compute_accuracy(self, testData_X, testData_Y):
        output = self.predict(testData_X)
        correct = 0
        for i in range(testData_X.shape[0]):
            if testData_Y[i] == output[i]:
                correct+=1
        accuracy = correct*100/testData_X.shape[0]
        return accuracy, output

def getConfusionMatrix(predicted, actual):
    confusion = np.zeros((10, 10), dtype=np.int32)  #[actual, predicted]
    for i in range(len(predicted)):
        confusion[actual[i], predicted[i]]+=1
    return confusion


def partF(trainingData, testData):
    # start = time.time()
    # model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver = 'sgd', learning_rate='adaptive', learning_rate_init=0.01, batch_size=100 ,max_iter=1000,random_state=2,shuffle=True )
    # model.fit(trainingData[0], trainingData[1])
    # end = time.time()
    # print("On Using Sigmoid")
    # print("Time taken: ", end-start)
    # train_accuracy = model.score(trainingData[0], trainingData[1])
    # test_accuracy = model.score(testData[0], testData[1])
    # print("Accuracy on train data: ", train_accuracy)
    # print("Accuracy on test data: ", test_accuracy)

    print()
    start = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver = 'sgd', learning_rate='adaptive', learning_rate_init=0.01, batch_size=100 ,max_iter=1000,random_state=2,shuffle=True )
    model.fit(trainingData[0], trainingData[1])
    end = time.time()
    print("On Using relu")
    print("Time taken: ", end-start)
    train_accuracy = model.score(trainingData[0], trainingData[1])
    test_accuracy = model.score(testData[0], testData[1])
    print("Accuracy on train data: ", train_accuracy)
    print("Accuracy on test data: ", test_accuracy)






def partE(trainingData, testData, rawTrainY, rawTestY):  
    print("On using sigmoid")
    model = neuralNetwork([100, 100], 100, 85, 10)
    model.train(trainingData[0], trainingData[1], 0.1, adaptive = True)
    trainAcc, output_train = model.compute_accuracy(trainingData[0], rawTrainY)
    print("Accuracy on train data: ", trainAcc)
    testAcc, output_test = model.compute_accuracy(testData[0], rawTestY)
    print("Accuracy on test data: ", testAcc)
    print(getConfusionMatrix(output_test, rawTestY))


    print()
    print("On Using Relu")
    model_relu = neuralNetwork([100, 100], 100, 85, 10, "relu")
    model_relu.train(trainingData[0], trainingData[1], 0.1, adaptive = True)
    trainAcc_relu, output_train = model_relu.compute_accuracy(trainingData[0], rawTrainY)
    print("Accuracy on train data: ", trainAcc_relu)
    testAcc_relu, output_test = model_relu.compute_accuracy(testData[0], rawTestY)
    print("Accuracy on test data: ", testAcc_relu)
    print(getConfusionMatrix(output_test, rawTestY))

   


def partD(trainingData, testData, rawTrainY, rawTestY):
    layerUnits = [5, 10, 15, 20, 25]
    timeTaken = []
    testAcuuracy = []
    trainAccuracy = []
    for unit in layerUnits:
        model = neuralNetwork([unit], 100, 85, 10)
        start = time.time()
        model.train(trainingData[0], trainingData[1], 0.1, adaptive=True)
        end = time.time()
        testAcc, outputTest = model.compute_accuracy(testData[0], rawTestY)
        print()
        print("No. of units in hidden layer: ", unit)
        print(getConfusionMatrix(outputTest, rawTestY))
        trainAcc, outputTrain = model.compute_accuracy(trainingData[0], rawTrainY)
        print("Time-taken: ", end-start)
        print("Test Accuracy: ", testAcc)
        print("Training Accuracy: ", trainAcc)
        timeTaken.append(end-start)
        testAcuuracy.append(testAcc)
        trainAccuracy.append(trainAcc)
        fig, ax = plt.subplots()

    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Time taken")
    ax.set_title("Time taken to train the model")
    ax.plot(layerUnits, timeTaken)
    plt.savefig("unit_vs_time_ad.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on test data")
    ax.plot(layerUnits, testAcuuracy)
    plt.savefig("unit_vs_testAccuracy_ad.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on train data")
    ax.plot(layerUnits, trainAccuracy)
    plt.savefig("unit_vs_trainAccuracy_ad.png")
    plt.show()



def partC(trainingData, testData, rawTrainY, rawTestY):
    layerUnits = [5, 10, 15, 20, 25]
    timeTaken = []
    testAccuracy = []
    trainAccuracy = []
    for unit in layerUnits:
        model = neuralNetwork([unit], 100, 85, 10)
        start = time.time()
        model.train(trainingData[0], trainingData[1], 0.1)
        end = time.time()
        testAcc, outputTest = model.compute_accuracy(testData[0], rawTestY)
        print()
        print("No. of units in hidden layer: ", unit)
        print(getConfusionMatrix(outputTest, rawTestY))
        trainAcc, outputTrain = model.compute_accuracy(trainingData[0], rawTrainY)
        print("Time-taken: ", end-start)
        print("Test Accuracy: ", testAcc)
        print("Training Accuracy: ", trainAcc)
        timeTaken.append(end-start)
        testAccuracy.append(testAcc)
        trainAccuracy.append(trainAcc)
    

    fig, ax = plt.subplots()
    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Time taken")
    ax.set_title("Time taken to train the model")
    ax.plot(layerUnits, timeTaken)
    plt.savefig("unit_vs_time.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on test data")
    ax.plot(layerUnits, testAccuracy)
    plt.savefig("unit_vs_testAccuracy.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("Units in hidden layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on train data")
    ax.plot(layerUnits, trainAccuracy)
    plt.savefig("unit_vs_trainAccuracy.png")
    plt.show()



def partB(trainData, testData, rawTrainY, rawTestY):
    model = neuralNetwork([5], 100, 85, 10)
    model.train(trainData[0], trainData[1], learningRate = 0.1)
    trainAccuracy, output = model.compute_accuracy(trainData[0], rawTrainY)
    print("Accuracy on training data: ", trainAccuracy)
    testAccuracy, output = model.compute_accuracy(testData[0], rawTestY)
    print("Accuracy on test data: ", testAccuracy)



def partA(trainData, testData):
    print("One-hot-encoding completed")
    print("Training Data (X) Shape: ", trainData[0].shape)
    print("Training Data (Y) Shape: ", trainData[1].shape)
    print("Test Data (X) Shape: ", testData[0].shape)
    print("Test Data (Y) Shape: ", testData[1].shape)



if __name__ == '__main__':
    path_train_data, path_test_data, part = sys.argv[1:]
    rawTrainData = pd.read_csv(path_train_data, sep=',', header = None).values
    rawTestData = pd.read_csv(path_test_data, sep =',', header = None).values
    trainDataX, trainDataY = hot_encoded_poker_data(rawTrainData)
    testDataX, testDataY = hot_encoded_poker_data(rawTestData)
    if part == 'a':
        partA((trainDataX, trainDataY), (testDataX, testDataY))
    elif part == 'b':
        partB((trainDataX, trainDataY), (testDataX, testDataY), rawTrainData[:, -1], rawTestData[:, -1])
    elif part == 'c':
        partC((trainDataX, trainDataY), (testDataX, testDataY), rawTrainData[:, -1], rawTestData[:, -1])
    elif part == 'd':
        partD((trainDataX, trainDataY), (testDataX, testDataY), rawTrainData[:, -1], rawTestData[:, -1])
    elif part == 'e':
        partE((trainDataX, trainDataY), (testDataX, testDataY), rawTrainData[:, -1], rawTestData[:, -1])
    elif part == 'f':
        partF((trainDataX, trainDataY), (testDataX, testDataY))
    