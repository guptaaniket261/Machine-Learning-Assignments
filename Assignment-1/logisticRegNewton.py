import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches


#function to normalise the dataset with mean = 0 and cov =1
def normalise(datasetX):
    mean_ar = np.mean(datasetX, axis = 0, keepdims=True)
    std_ar = np.std(datasetX, axis = 0, keepdims=True)
    normalisedData = (datasetX - mean_ar)/std_ar
    return (mean_ar, std_ar, normalisedData)

#return the sigmoid value
def sigmoid(datasetX):
    sol = 1/(1+np.exp(-datasetX))
    return sol

#returns log likelikelihood at a particular theta
def logLikelihood(datasetX, datasetY, theta):
    m = datasetX.shape[0]
    hyp = np.reshape(sigmoid(datasetX.dot(theta)), (m, 1))
    llh = np.sum(datasetY*np.log(hyp) + (1-datasetY)*np.log(1-hyp))
    return llh

#returns the hessian matrix for the given data at particular theta
def getHessian(datasetX, datasetY, theta):
    m = datasetX.shape[0]
    hyp = np.reshape(sigmoid(datasetX.dot(theta)), (m, 1))
    dh = np.diag((hyp*(1-hyp))[:, 0])
    hessian = (datasetX.T).dot(dh).dot(datasetX)
    return -1*hessian

#return gradient of the log likelihood function at a given theta value
def getGrad(datasetX, datasetY, theta):
    m = datasetX.shape[0]
    hyp = np.reshape(sigmoid(datasetX.dot(theta)), (m, 1))
    grad = (datasetX.T).dot(datasetY-hyp)
    return grad


# main function to learn parameters using newton method
def newtonMethod(datasetX, datasetY):
    eps = 0.01 # used to check convergence
    converged = False
    theta = np.zeros((3,1))  #initialise parameters to 0 vector
    # storing the values for later analysis
    old_llh = logLikelihood(datasetX, datasetY, theta)
    llh_ar = [old_llh]
    iter_ar = [0]
    count = 0
    while(not(converged)):
        # get the hessian matrix
        hessian = getHessian(datasetX, datasetY, theta)
        #get the gradient vector
        grad = getGrad(datasetX, datasetY, theta)
        #uodate the parameter
        theta = theta - np.linalg.inv(hessian).dot(grad)
        new_llh = logLikelihood(datasetX, datasetY, theta)
        #check for convergence
        if(abs(new_llh - old_llh)<eps):
            converged = True
        count+=1
        iter_ar.append(count)
        llh_ar.append(new_llh)
        old_llh = new_llh

    #plot log likelihood vs no of parameters
    print(llh_ar)
    fig,ax = plt.subplots()
    ax.set_xlabel("No. of iterations")
    ax.set_ylabel("Log likelihood")
    ax.set_title("Variation of log likelihood function with no. of iterations")
    ax.plot(iter_ar, llh_ar)
    plt.show()
    return theta

#function to plot the separator obatined
def plotDataSeparator(datasetXi, datasetY, mean_arr, std_arr, theta):
    m, n = datasetXi.shape
    fig,ax = plt.subplots()
    colors = np.array(['red','green'])
    dY = np.array([int(datasetY[i, 0]) for i in range(m)])
    x1 = [min(datasetXi[:,1]), max(datasetXi[:,1])]
    const = theta[1,0]*mean_arr[0,0]/std_arr[0,0] + theta[2,0]*mean_arr[0,1]/std_arr[0,1]
    x2 = [(std_arr[0,1]/theta[2,0])*(-theta[0,0]-(theta[1,0]*x1[0])/std_arr[0,0]+ const), (std_arr[0,1]/theta[2,0])*(-theta[0,0]-(theta[1,0]*x1[1])/std_arr[0,0]+ const)]
    ax.scatter(datasetXi[:, 0], datasetXi[:, 1], c = colors[dY])
    r_patch = mpatches.Patch(color = 'red', label = 'Class 0')
    g_patch = mpatches.Patch(color = 'green', label = 'Class 1')
    ax.legend(handles = [r_patch, g_patch], loc = "upper right")
    ax.plot(x1, x2)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Separator obtained using Newton Method')
    plt.show()

#read input file names using terminal
args = sys.argv[1:]
datasetXfile, datasetYfile = args[0], args[1]
datasetXi = pd.read_csv(datasetXfile, header=None).values
datasetY = pd.read_csv(datasetYfile, header = None).values

mean_arr, std_arr, datasetX = normalise(datasetXi)
m, n = datasetX.shape
intercept_col = np.ones((m,1))
datasetX = np.concatenate((intercept_col, datasetX), axis = 1)


theta = newtonMethod(datasetX, datasetY)
print(theta)
plotDataSeparator(datasetXi, datasetY, mean_arr, std_arr, theta)







    
    
