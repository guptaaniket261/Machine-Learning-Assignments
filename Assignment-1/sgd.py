import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


# function to sample data points following normal distribution with given means and covariance
def sampleDataPoints(theta, mean_x1, cov_x1, mean_x2, cov_x2, cov_error, size):
    x1 = np.reshape(np.random.normal(mean_x1, cov_x1**0.5, size), (size,1))
    x2 = np.reshape(np.random.normal(mean_x2, cov_x2**0.5, size), (size,1))
    error = np.reshape(np.random.normal(0, cov_error**0.5, size), (size,1))
    x0 = np.ones((size, 1), dtype= np.float64)
    datasetX = np.concatenate((x0, x1, x2), axis = 1)
    hypothesis = datasetX.dot(theta)
    datasetY = hypothesis + error
    return (datasetX, datasetY)

#function to return loss function over a given batch for particular value of theta
def getLoss(batchX, batchY, theta):
    hyp = batchX.dot(theta)
    sq_error_i = np.square(hyp-batchY)
    loss = np.sum(sq_error_i)/(2*batchX.shape[0])
    return loss

#function to get gradient of the loss function at given value of theta for a given batch
def grad(batchX, batchY, theta):
    error_i = batchX.dot(theta) - batchY
    grad_arr = (batchX.T).dot(error_i)/(batchX.shape[0])
    return grad_arr

# main function to implement stochastic gradient descent
def sgd(datasetX, datasetY, learning_rate, batchsize, k, delta):
    start = time.time()
    count = 0
    theta = np.zeros((3,1))  #initialise the parameter to zero vector
    converged = False
    # arrays to store data for further analysis
    iter_ar = []
    loss_ar = []
    st_index = 0
    old_loss = 0
    theta_ar = [[0,0,0]]
    while(converged == False):
        # determine the current batch of data
        end_index = st_index+batchsize
        batch_X = datasetX[st_index:end_index, :]
        batch_Y = datasetY[st_index:end_index, :]

        #update the value of parameter
        theta = theta - learning_rate*grad(batch_X, batch_Y, theta)
        #store the values for further analysis
        theta_ar.append([theta[0,0], theta[1,0], theta[2,0]])
        loss = getLoss(batch_X, batch_Y, theta)
        count+=1
        loss_ar.append(loss)
        iter_ar.append(count)

        # check for convergence
        if(count%k == 0):
            new_loss = np.sum(loss_ar[-k:])/k
            if(abs(old_loss-new_loss)<delta):
                converged = True
            old_loss = new_loss
        st_index+=batchsize
        if(st_index+batchsize>datasetX.shape[0]):
            st_index = 0
        '''if(count>10000):
            break'''
    endTime = time.time()

    #print basic analysis of the algorithm used 
    print("learning rate: {0}, batch size: {1}, k: {2}, delta: {3}".format(learning_rate,batchSize, k, delta))
    print("Theta: ", theta)
    print("Iterations: ", count)
    print("Time taken: ",(endTime-start) )

    # plot the loss function vs parameter
    fig,ax = plt.subplots()
    ax.plot(iter_ar, loss_ar)
    ax.set_title("Loss function vs Iterations, Batch Size = {0}, k = {1}".format(batchSize, k))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss function")
    plt.savefig("LossVsIter_batch{0}.png".format(batchSize))
    plt.show()
    
    #return the parameters learned
    return theta, np.array(theta_ar)

#function to test on a particular test file 
def test(testFile, theta):
    dataset = pd.read_csv(testFile).values
    dataX = dataset[:,0:2]
    size = dataset.shape[0]
    x0 = np.ones((size, 1), dtype= np.float64)
    dataX = np.concatenate((x0, dataX), axis = 1)
    dataY = dataset[:, 2].reshape((size,1))
    print(dataX, dataY)
    error = getLoss(dataX, dataY, theta)
    print(theta)
    print(error)

# function to plot the movement of parameter in 3D during the parameter learning
def plotParameterMovement(theta_ar, batchSize):
    ax = plt.axes(projection = '3d')
    ax.plot3D(theta_ar[:,0], theta_ar[:,1], theta_ar[:,2])
    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_zlabel("Theta_2")
    ax.set_title("Movement of theta before convergence, Batch Size = {0}".format(batchSize))
    plt.show()


#read filename and other parameters from terminal
args = sys.argv[1:]
learning_rate, batchSize, k, delta = float(args[0]), int(args[1]), int(args[2]), float(args[3])
datasetX, datasetY = sampleDataPoints(np.array([[3],[1],[2]]), 3,4,-1,4,2, 1000000)
theta, theta_ar = sgd(datasetX, datasetY, learning_rate, batchSize, k, delta)  ##sgd(datasetX, datasetY, learning_rate, batchsize, k, delta)
print(theta)
plotParameterMovement(theta_ar, batchSize)

#theta_test = np.array([2.88471293, 1.0259371, 1.99168564]).reshape((3,1))
#test('data/q2/q2test.csv', theta_test)



