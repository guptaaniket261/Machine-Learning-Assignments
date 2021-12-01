from numpy import linalg
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys
from cvxopt import matrix
from cvxopt import solvers
import time
from PIL import Image

trainingData = {}
testData = {}
mod_no = {}

# predict labels for dataset X using given model
def predict(X, model, para = 0.05):
    alpha_sv, sv, sv_y, b = model["alpha_sv"], model["sv"], model["sv_y"], model["b"]
    normSqX = np.linalg.norm(X, axis = 1, keepdims=True)**2
    expNormSqX = np.exp(-1*para*normSqX)

    normSqSV = np.linalg.norm(sv, axis = 1, keepdims=True)**2
    expNormSqSV = np.exp(-1*para*normSqSV)
    dotMat = X.dot(sv.T)
    expDotMat = np.exp(2*para*dotMat)
    #print(expNormSqX.shape,expNormSqSV.shape, expDotMat.shape )
    predicted_y = expNormSqX*(expDotMat.dot(sv_y*alpha_sv*expNormSqSV)) + b
    #print(predicted_y.shape)
    return predicted_y

# additional function to test a particular model
def test_one_model(dataset, model):
    m,n = dataset.shape
    X = dataset[:, :(n-1)]
    Y = dataset[:, n-1].reshape(m,)
    predicted_y = predict(X, model)
    correct = 0
    wrong = []
    corr = []
    for i in range(X.shape[0]):
        if(predicted_y[i,0]*Y[i]>=0):
            correct +=1
            corr.append((i,predicted_y[i,0],Y[i]))
        else:
            wrong.append((i,predicted_y[i,0],Y[i]))
    print("Accuracy: ")
    print(correct*100/X.shape[0])

# function to draw and save images
def drawImage(imgPixels,img_name):
    img = imgPixels.reshape((28, 28))
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.save(img_name)

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
                drawImage(testData[j,:784], "OvsO_{0}_predictedAs_{1}.png".format(i[0], i[1]))

# get a confusion matrix for given outcomes
def getConfusionMatrix(predicted, actual):
    confusion = np.zeros((10, 10), dtype=np.int32)  #[actual, predicted]
    for i in range(len(predicted)):
        confusion[int(actual[i]), int(predicted[i])]+=1
    return confusion

# test on test-set using the given model
def test(models):
    m,n = testData.shape
    votes = np.zeros((m,10))
    data_test = testData[:, :(n-1)]/255
    models_predictions = []
    prediction = []
    for i in range(10):
        for j in range(i+1, 10):
            curr_mod = models[(i,j)]
            curr_pred = predict(data_test, curr_mod)
            models_predictions.append(curr_pred.reshape((m,)))
    for i in range(m):
        for a in range(10):
            for b in range(a+1, 10):
                if models_predictions[mod_no[(a,b)]][i]>0:
                    votes[i,a] += 1
                else:
                    votes[i,b] += 1
        pred_i = 0
        max_vote = -1
        for j in range(10):
            if votes[i, j] > max_vote:
                pred_i = j
                max_vote = votes[i, j]
            elif votes[i, j] == max_vote and models_predictions[mod_no[(pred_i, j)]][i]<0:
                pred_i = j
                max_vote = votes[i, j]
        prediction.append(pred_i)
    actual = testData[:, n-1].reshape((m,))
    correct = 0
    for i in range(m):
        if actual[i] == prediction[i]:
            correct+=1
    accuracy = correct/m
    print()
    print("Accuracy on multiclass one-vs-one SVM classifier: ", accuracy*100)
    return (prediction, actual)

# get gaussian kernel matrix for given dataset X
def getKernelMat(X, para = 0.05):
    normX = np.linalg.norm(X, axis = 1 , keepdims=True)
    normSq = normX*normX
    expNormSq = np.exp(-1*para*normSq)
    dotMat = np.exp(2*para*(X.dot(X.T)))
    kernelMat = expNormSq*dotMat*(expNormSq.T)
    return kernelMat


# train a SVM model using gaussian kernel
def gaussSVM(dataset, C = 1):
    m,n = dataset.shape
    X = dataset[:, :(n-1)]
    Y = dataset[:, n-1].reshape((-1,1))  

    kernel = getKernelMat(X)
    P = matrix(np.outer(Y,Y)*kernel, tc = 'd')
    q = matrix(-1*np.ones((m,1)), tc = 'd')
    diag_one = np.diag((np.ones((m,))))
    G = matrix(np.concatenate((diag_one, -1*diag_one), axis = 0), tc = 'd')
    h = matrix(np.vstack((np.array([C for i in range(m)]).reshape(-1,1), np.zeros((m,1)))), tc = 'd')
    A = matrix(Y.reshape(1,-1))
    b = matrix(np.zeros((1,1), dtype=np.float64))

    #solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h, A, b)
    alpha = np.array(sol['x'])
    threshold = 1e-14
    S = (alpha>threshold).flatten()
    alpha_sv, sv, sv_y = alpha[S], X[S], Y[S]
    model = {"sv": sv, "sv_y":sv_y, "alpha_sv":alpha_sv, "b":0}
    sv_pred = predict(sv, model)
    b = 0
    mx, mn = -np.inf, np.inf
    count = 0
    for i in range(alpha_sv.shape[0]):
        val = sv_y[i] - sv_pred[i]
        if(abs(alpha_sv[i]-C)<threshold):
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
    model["b"] = b
    return (model)

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


# main function to handle model training,testing and result analysis
def main(trainFile, testFile, part_c=0):
    global trainingData
    global testData
    global mod_no
    trainingData = pd.read_csv(trainFile, header = None).values
    testData = pd.read_csv(testFile, header = None).values

    _,n = trainingData.shape
    dataset_digit = {}
    for dig in range(10):
        dataset_digit[dig] = getData(dig)
        dataset_digit[dig] = dataset_digit[dig][:, :(n-1)]
    models = {}
    count = 0
    startTime = time.time()
    for i in range(10):
        one = np.ones((dataset_digit[i].shape[0],1))
        for j in range(i+1,10):
            mod_no[(i,j)] = count
            count+=1
            neg = -1*np.ones((dataset_digit[j].shape[0],1))
            data = np.vstack((np.hstack((dataset_digit[i],one)), np.hstack((dataset_digit[j], neg))))
            print()
            print("Training model for lables:", (i,j))
            models[(i, j)] = gaussSVM(data)
            #test_one_model(data, models[(i,j)]
    endTime = time.time()
    training_time = endTime - startTime
    print("Time taken to train multiclass one-vs-one SVM classifier: ", training_time)
    predicted, actual = test(models)
    if part_c==1:
        conf = getConfusionMatrix(predicted, actual)
        print("Confusion Matrix Obtained:")
        print(conf)
        #getMissClassifiedImages(conf, predicted, actual)



# entry point in the code
if __name__ == '__main__':
    arg = sys.argv[1:]
    trainFile, testFile, part_c = arg[0], arg[1], arg[2]
    #part_c = 1 for part c
    main(trainFile, testFile, int(part_c))