from numpy import linalg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches



# 0 - alaska, 1 - canada
# give labels 0 or 1 to y value
def getLabel(datasetY):
    m = datasetY.shape[0]
    label = np.zeros((m,1))
    for i in range(m):
        if (datasetY[i,0].strip() == "Canada"):
            label[i,0] = 1
    return label

#preprocess the input data, change from string to float
def preprocessX(datasetXi):
    datasetX = np.zeros((datasetXi.shape[0],2))
    for i in range(datasetXi.shape[0]):
        temp = [float(j) for j in datasetXi[i, 0].strip().split()]
        datasetX[i,0] = temp[0]
        datasetX[i,1] = temp[1]
    return datasetX

# normalise the given dataset
def normalise(datasetX):
    mean_ar = np.mean(datasetX, axis = 0, keepdims=True)
    std_ar = np.std(datasetX, axis = 0, keepdims=True)
    normalisedData = (datasetX - mean_ar)/std_ar
    return (mean_ar, std_ar, normalisedData)

#implement gda with same covariance for both classes
def paramA(datasetX, datasetY):
    n = datasetX.shape[1]
    m = datasetX.shape[0]
    mean0 = np.zeros((1,n))
    count0 = 0
    count1 = 0
    mean1 = np.zeros((1,n))
    covariance = np.zeros((n,n))

    #calculate mean vectors for both classes
    for i in range(m):
        currTrainingData = np.reshape(datasetX[i, :], (1,n))
        if(datasetY[i,0] == 0):
            mean0 = mean0 + currTrainingData
            count0 +=1
        else:
            mean1 = mean1 + currTrainingData
            count1 +=1
    mean0/=count0
    mean1/=count1
    #calculate covariance matrix
    for i in range(m):
        currTrainingData = np.reshape(datasetX[i, :], (1,n))
        '''if(i==0):
            print(currTrainingData)
            print(mean0)
            print(currTrainingData-mean0)
            print(np.dot((currTrainingData-mean0).T, (currTrainingData-mean0)))'''
        if(datasetY[i,0] == 0):
            covariance = covariance + np.dot((currTrainingData-mean0).T, (currTrainingData-mean0))
        else:
            covariance = covariance + np.dot((currTrainingData-mean1).T, (currTrainingData-mean1))
    covariance/=m

    mean0 = mean0.T
    mean1 = mean1.T
    print(mean0, mean1, covariance)
    return (mean0, mean1, covariance)

# implement gda with different covariances
def paramB(datasetX, datasetY):
    n = datasetX.shape[1]
    m = datasetX.shape[0]
    mean0 = np.zeros((1,n))
    count0 = 0
    count1 = 0
    mean1 = np.zeros((1,n))
    covariance0 = np.zeros((n,n))
    covariance1 = np.zeros((n,n))

    for i in range(m):
        currTrainingData = np.reshape(datasetX[i, :], (1,n))
        if(datasetY[i,0] == 0):
            mean0 = mean0 + currTrainingData
            count0 +=1
        else:
            mean1 = mean1 + currTrainingData
            count1 +=1
    mean0/=count0
    mean1/=count1

    for i in range(m):
        currTrainingData = np.reshape(datasetX[i, :], (1,n))
        '''if(i==0):
            print(currTrainingData)
            print(mean0)
            print(currTrainingData-mean0)
            print(np.dot((currTrainingData-mean0).T, (currTrainingData-mean0)))'''
        if(datasetY[i,0] == 0):
            covariance0 = covariance0 + np.dot((currTrainingData-mean0).T, (currTrainingData-mean0))
        else:
            covariance1 = covariance1 + np.dot((currTrainingData-mean1).T, (currTrainingData-mean1))
    covariance0/=count0
    covariance1/=count1

    mean0 = mean0.T
    mean1 = mean1.T
    print(mean0, mean1, covariance0, covariance1)
    return (mean0, mean1, covariance0, covariance1)

# plot linear separator
def plotSeparatorA(datasetX, datasetY, mean0, mean1, covariance):
    fig,ax = plt.subplots()
    colors = np.array(['red','green'])
    dY = np.array([int(datasetY[i, 0]) for i in range(datasetY.shape[0])])
    ax.scatter(datasetX[:, 0], datasetX[:, 1], c = colors[dY])
    x = [min(datasetX[:,0]), max(datasetX[:,1])]
    m = datasetY.shape[0]
    phi = 0
    for i in range(m):
        phi+=datasetY[i,0]
    phi/=m
    covInv = np.linalg.inv(covariance)
    const = np.log((1-phi)/phi) + ((mean1.T).dot(covInv).dot(mean1) - (mean0.T).dot(covInv).dot(mean0))/2
    const = const[0,0]
    a = ((mean1-mean0).T).dot(covInv)
    y = [(const - x[0]*a[0,0])/a[0,1], (const - x[1]*a[0,0])/a[0,1]]
    ax.plot(x,y)
    ax.set_xlabel("Feature X1")
    ax.set_ylabel("Feature X2")
    ax.set_title("Plot of the data and linear separator")
    r_patch = mpatches.Patch(color = 'red', label = 'Alaska')
    g_patch = mpatches.Patch(color = 'green', label = 'Canada')
    ax.legend(handles = [r_patch, g_patch], loc = "upper right")
    plt.show()

#get quadratic separator value
def quadSeparator(phi, mean0, mean1, covariance0, covariance1, x):
    m = datasetY.shape[0]
    sqrtDetCov0 = np.sqrt(np.linalg.det(covariance0))
    sqrtDetCov1 = np.sqrt(np.linalg.det(covariance1))
    covInv0 = np.linalg.inv(covariance0)
    covInv1 = np.linalg.inv(covariance1)
    const = np.log(((1-phi)/phi)*(sqrtDetCov1/sqrtDetCov0)) + ((mean1.T).dot(covInv1).dot(mean1) - (mean0.T).dot(covInv0).dot(mean0))/2
    nonConst = (x.T).dot(covInv1-covInv0).dot(x)/2 - ((mean1.T).dot(covInv1) - (mean0.T).dot(covInv0)).dot(x)
    sol = nonConst + const
    return sol[0,0]

#get phi value for the given data
def getPhi(datasetY):
    m = datasetY.shape[0]
    phi = 0
    for i in range(m):
        phi+=datasetY[i,0]
    phi/=m
    return phi

#plot quadratic separator with given parameters
def plotSepartorB(datasetX, datasetY, mean0, mean1, covariance0, covariance1):
    fig,ax = plt.subplots()
    colors = np.array(['red','green'])
    dY = np.array([int(datasetY[i, 0]) for i in range(datasetY.shape[0])])
    ax.scatter(datasetX[:, 0], datasetX[:, 1], c = colors[dY])
    x = [min(datasetX[:,0]), max(datasetX[:,1])]
    m = datasetY.shape[0]
    phi = getPhi(datasetY)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x,y, indexing = 'xy')
    z = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(z):
        z[i,j] = quadSeparator(phi, mean0, mean1, covariance0, covariance1, np.array([[X[i,j]],[Y[i,j]]]))
    ax.contour(X, Y, z, 0)
    ax.set_xlabel("Feature X1")
    ax.set_ylabel("Feature X2")
    ax.set_title("Plot of the data and Quadratic separator")
    r_patch = mpatches.Patch(color = 'red', label = 'Alaska')
    g_patch = mpatches.Patch(color = 'green', label = 'Canada')
    ax.legend(handles = [r_patch, g_patch], loc = "upper right")
    plt.show()


#take input from terminal
args = sys.argv[1:]
datasetXfile, datasetYfile = args[0], args[1]
datasetXi = pd.read_csv(datasetXfile, header=None).values
datasetYi = pd.read_csv(datasetYfile, header = None).values

datasetX = preprocessX(datasetXi)
datasetY = getLabel(datasetYi)

'''fig,ax = plt.subplots()
colors = np.array(['red','green'])
dY = np.array([int(datasetY[i, 0]) for i in range(datasetY.shape[0])])
ax.scatter(datasetX[:, 0], datasetX[:, 1], c = colors[dY])
ax.set_xlabel("Feature X1")
ax.set_ylabel("Feature X2")
ax.set_title("Plot of the data")
r_patch = mpatches.Patch(color = 'red', label = 'Alaska')
g_patch = mpatches.Patch(color = 'green', label = 'Canada')
ax.legend(handles = [r_patch, g_patch], loc = "upper right")
plt.show()'''

mean_arr, std_arr, datasetX = normalise(datasetX)

mean0, mean1, covariance = paramA(datasetX, datasetY)
mean0, mean1, covariance0, covariance1 = paramB(datasetX, datasetY)


#red , up- alaska
plotSeparatorA(datasetX,datasetY, mean0, mean1, covariance)
plotSepartorB(datasetX, datasetY, mean0, mean1, covariance0, covariance1)

