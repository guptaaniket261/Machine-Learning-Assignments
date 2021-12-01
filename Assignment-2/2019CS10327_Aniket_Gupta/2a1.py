from numpy import linalg
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys
from cvxopt import matrix
from cvxopt import solvers
import time

trainingData = {}
testData = {}

#get data for a particular file
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

# training a SVM model using a linear kernel
def linearSVM(dataset, C = 1):
    m,n = dataset.shape
    X = dataset[:, :(n-1)]
    Y = dataset[:, n-1].reshape((-1,1))

    Y_X = Y*X
    P = matrix(np.dot(Y_X, Y_X.T))
    q = matrix(-1*np.ones((m,1)))
    diag_one = np.diag((np.ones((m,))))
    G = matrix(np.vstack((diag_one, -1*diag_one)), tc = 'd')
    h = matrix(np.vstack((np.array([C for i in range(m)]).reshape(-1,1), np.zeros((m,1)))), tc = 'd')
    A = matrix(Y.reshape(1,-1))
    b = matrix(np.zeros((1,1), dtype=np.float64))

    sol = solvers.qp(P,q,G,h, A, b)

    alpha = np.array(sol['x'])
    w = (((Y*alpha).T).dot(X)).reshape((-1,1))
    threshold = 1e-6
    S = (alpha>threshold).flatten()
    alpha_sv, sv, sv_y = alpha[S], X[S], Y[S]
    print()
    print("No. of Support Vectors:",alpha_sv.shape[0])
    b = 0
    mx, mn = -np.inf, np.inf
    count = 0
    for i in range(alpha_sv.shape[0]):
        val = sv_y[i] - w.T.dot(sv[i,:])
        if(abs(alpha_sv[i]-C)<1e-6):
            if(int(sv_y[i])==-1):
                mn = min(mn, val)
            else:
                mx = max(mx, val)
            continue
        count += 1
        b += val
    if count!=0:
        b/= count
    else:
        b = (mx+mn)/2
    print("bias (b):", b)
    return (w, b)

# testing the model on the given dataset
def test(dataset, model):
    m,n = dataset.shape
    X = dataset[:, :(n-1)]
    Y = dataset[:, n-1].reshape(m,)
    w, b = model
    predicted_y = []
    correct = 0
    for i in range(X.shape[0]):
        p_y = w.T.dot(X[i, :]) + b
        predicted_y.append(p_y)
        if(p_y*Y[i]>0):
            correct +=1
    print(correct*100/X.shape[0])


# main function to handle model training,testing and result analysis
def main(trainFile, testFile, digit1=7, digit2=8):
    global trainingData
    global testData
    trainingData = pd.read_csv(trainFile, header = None).values
    testData = pd.read_csv(testFile, header = None).values

    dig1Data = getData(digit1)
    dig2Data = getData(digit2)
    m1, n1 = dig1Data.shape
    m2, n2 = dig2Data.shape
    dig1Data = np.hstack((dig1Data[:, :(n1-1)], np.ones((m1,1))))
    dig2Data = np.hstack((dig2Data[:, :(n2-1)], -1*np.ones((m2,1))))
    data = np.vstack((dig1Data, dig2Data))
    print()
    print("Training...")
    start = time.time()
    model = linearSVM(data, C=1)
    end = time.time()
    print("Time taken to train the model (in sec):", end-start)
    print()
    print("Testing...")
    dig1Data_test = getData(digit1, False)
    dig2Data_test = getData(digit2, False)
    m1, n1 = dig1Data_test.shape
    m2, n2 = dig2Data_test.shape
    dig1Data_test = np.hstack((dig1Data_test[:, :(n1-1)], np.ones((m1,1))))
    dig2Data_test = np.hstack((dig2Data_test[:, :(n2-1)], -1*np.ones((m2,1))))
    data_test = np.vstack((dig1Data_test, dig2Data_test))
    print()
    print("Accuracy on test data: ")
    test(data_test, model)
    #test(data, model)

# entry point for the function
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile = arg[0], arg[1]
    main(trainFile, testFile, 7,8)
