from libsvm.svmutil import svm_predict, svm_train
from numpy import linalg
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys
from libsvm.svmutil import *
import time

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

# train SVM model using a linear kernel
def getLinearParams(sv_coeff_linear, dataX, labels, sv_indices, C=1):
    nsv = len(sv_coeff_linear)
    sv = dataX[sv_indices]
    sv_alpha = np.array([sv_coeff_linear[i][0] for i in range(nsv)]).reshape((nsv,1))
    w = sv_alpha.T.dot(sv).reshape((-1,1))
    b = 0
    mx, mn = -np.inf, np.inf
    count = 0
    for i in range(nsv):
        label = labels[sv_indices[i]]
        val =  label - w.T.dot(sv[i,:])
        if(abs(abs(sv_alpha[i])-C)<1e-7):
            if(int(label)==-1):
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
    return w,b

# predict labels on dataset X using model 
def predict(X, model, para = 0.05):
    alpha_sv, sv, sv_y, b = model["alpha_sv"], model["sv"], model["sv_y"], model["b"]
    normSqX = np.linalg.norm(X, axis = 1, keepdims=True)**2
    expNormSqX = np.exp(-1*para*normSqX)

    normSqSV = np.linalg.norm(sv, axis = 1, keepdims=True)**2
    expNormSqSV = np.exp(-1*para*normSqSV)
    dotMat = X.dot(sv.T)
    expDotMat = np.exp(2*para*dotMat)
    #print(expNormSqX.shape,expNormSqSV.shape, expDotMat.shape )
    predicted_y = expNormSqX*(expDotMat.dot(alpha_sv*expNormSqSV)) + b
    #print(predicted_y.shape)
    return predicted_y

# train the model with gaussian kernel
def getGaussParams(sv_coeff_gauss,dataX,dataY, sv_gauss_indices, C=1):
    nsv = len(sv_gauss_indices)
    sv = dataX[sv_gauss_indices]
    sv_alpha = np.array([sv_coeff_gauss[i][0] for i in range(nsv)]).reshape((nsv,1))
    sv_y = np.array([dataY[sv_gauss_indices[i]] for i in range(nsv)]).reshape((nsv,1))
    model = {"sv":sv, "alpha_sv":sv_alpha, "sv_y":sv_y, "b":0}
    sv_pred = predict(sv, model)
    b = 0
    mx, mn = -np.inf, np.inf
    count = 0
    for i in range(nsv):
        val = sv_y[i] - sv_pred[i]
        if(abs(sv_alpha[i]-C)<1e-6):
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
    return b

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
    
    dataX = np.vstack((dig1Data[:, :(n1-1)], dig2Data[:, :(n2-1)]))
    dataY = np.vstack((np.ones((m1,1)), -1*np.ones((m2,1)))).reshape(-1,)

    dig1Data_test = getData(digit1, False)
    dig2Data_test = getData(digit2, False)
    m1, n1 = dig1Data_test.shape
    m2, n2 = dig2Data_test.shape
    dataX_test = np.vstack((dig1Data_test[:, :(n1-1)], dig2Data_test[:, :(n2-1)]))
    dataY_test = np.vstack((np.ones((m1,1)), -1*np.ones((m2,1)))).reshape(-1,)
    print("\n\n")
    print("Using Linear Kernel: ")
    start = time.time()
    m_linear = svm_train(dataY, dataX, '-t 0 -c 1')
    end = time.time()
    print("Time taken to train the model using linear kernel (in sec):", end-start)
    print()
    svm_predict(dataY_test, dataX_test, m_linear)
    sv_linearKernel = m_linear.get_SV()
    sv_coeff_linear = m_linear.get_sv_coef()
    sv_indices = m_linear.get_sv_indices()
    
    NSV = len(sv_linearKernel)
    for i in range(NSV):
        sv_indices[i]-=1
    w, b = getLinearParams(sv_coeff_linear, dataX, dataY, sv_indices)

    print("\n\n")
    print("Using Gaussian Kernel: ")
    start = time.time()
    m_gaussian = svm_train(dataY, dataX, '-t 2 -c 1 -g 0.05')
    end = time.time()
    print("Time taken to train the model using gaussian kernel (in sec):", end-start)
    print()
    svm_predict(dataY_test, dataX_test, m_gaussian)
    
    sv_gaussKernel = m_gaussian.get_SV()
    sv_coeff_gauss = m_gaussian.get_sv_coef()
    sv_gauss_indices = m_gaussian.get_sv_indices()
    NSV = len(sv_gaussKernel)
    for i in range(NSV):
        sv_gauss_indices[i]-=1
    print("No. of Support Vectors:",len(sv_gauss_indices))
    b = getGaussParams(sv_coeff_gauss, dataX, dataY, sv_gauss_indices)
    
# enntry point in the code
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile = arg[0], arg[1]
    main(trainFile, testFile, 7,8)
    