import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys



#this function helps to normalise the given dataset with mean = 0 and cov = 1
def normalise(datasetX):
    mean_ar = np.mean(datasetX, axis = 0, keepdims=True)
    std_ar = np.std(datasetX, axis = 0, keepdims=True)
    normalisedData = (datasetX - mean_ar)/std_ar
    return (mean_ar, std_ar, normalisedData)

#this function returns loss/error function for a given value of theta on the given dataset
def getLoss(theta, normalisedData, datasetY):
    m = normalisedData.shape[0]
    hypothesis = normalisedData.dot(theta)
    squared_diff = np.square(datasetY - hypothesis)
    loss = np.sum(squared_diff)/(2*m)
    return loss


#This function returns the gradient of the loss function at any value of theta for the given dataset
def getGrad(normalisedData, datasetY, theta):
    m = normalisedData.shape[0]
    hypothesis = normalisedData.dot(theta)
    diff = hypothesis - datasetY
    grad = np.dot(normalisedData.T, diff)/m
    return grad
    
#This function implements the linear regression method to learn the parameters
def linearReg(normalisedData, datasetY, learning_rate = 0.01):
    m,n = normalisedData.shape
    theta = np.zeros((n,1))  #initialise theta
    delta = 0.0000000001   #used to check convergence
    converged = False   
    count = 0
    # these arrays store the parameter values for plotting
    iter_arr = [0]         
    prevLoss = getLoss(theta, normalisedData, datasetY)
    loss_ar = [prevLoss]
    theta_ar = [[0,0]]
    while(not(converged)):
        # updating the parameter
        theta = theta - learning_rate*getGrad(normalisedData, datasetY, theta)   
        newLoss = getLoss(theta, normalisedData, datasetY)
        count+=1
        iter_arr.append(count)
        theta_ar.append([theta[0,0], theta[1,0]])
        loss_ar.append(newLoss)
        if(abs(prevLoss-newLoss)<delta):
            converged = True
        prevLoss = newLoss
    #plt.plot(iter_arr, loss_ar)
    #returning the parameter learned
    return (theta, np.array(theta_ar), np.array(loss_ar))


# this function helps to draw the mesh 
def drawMesh(normalisedData, datasetY, theta_ar, loss_ar):
    x = np.linspace(0, 2, 100)
    y = np.linspace(-1, 1, 100)
    X_values, Y_values = np.meshgrid(x,y, indexing = 'xy')
    Z_values = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(Z_values):
        Z_values[i,j] = getLoss(np.array([[X_values[i,j]], [Y_values[i,j]]]), normalisedData, datasetY)
    ax = plt.axes(projection = '3d')
    theta0 = theta_ar[:, 0].reshape(theta_ar.shape[0],)
    theta1 = theta_ar[:, 1].reshape(theta_ar.shape[0],)
    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_zlabel("Error Value")
    ax.plot3D(theta0, theta1, loss_ar, label= "Error Values", color = 'red')

    ax.plot_surface(X_values, Y_values, Z_values, rstride = 1, cstride = 1, alpha = 0.6)
    ax.set_title("Error value as function of parameters")
    plt.savefig("ErrorValues.png")
    plt.show()


# draws animation showing the update of parameters
def drawAnimation(normalisedData, datasetY, theta_ar, loss_ar):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    x = np.linspace(0, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y, indexing = 'xy')
    z = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(z):
        z[i,j] = getLoss(np.array([[X[i,j]], [Y[i,j]]]), normalisedData, datasetY)
    ax.plot_surface(X, Y, z, rstride = 1, cstride = 1, alpha = 0.6)

    def update_line(num, data, line):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line

    data = np.array([theta_ar[:,0], theta_ar[:,1], loss_ar])
    line = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])[0]

    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_zlabel("Error Value")
    ax.set_title('Variation of error value with iterations')
    line_ani = animation.FuncAnimation(fig, update_line, loss_ar.size-10, fargs=(data, line), interval=200, blit=False)
    #line_ani.save('errorFunction.mp4', writer = 'ffmpeg', fps = 5)
    plt.show()

# plot contour for the given loss function
def plotContour(normalisedData, datasetY, loss_ar, theta_ar):
    fig,ax = plt.subplots()
    x = np.linspace(0, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y, indexing = 'xy')
    z = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(z):
        z[i,j] = getLoss(np.array([[X[i,j]], [Y[i,j]]]), normalisedData, datasetY)

    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_title("Movement of parameters on contour plot")
    i = 0
    while (i<loss_ar.size):
        ax.contour(X,Y,z, [loss_ar[i]], colors = ['gray'], alpha = 0.4)
        i+=1
    
    ax.plot(theta_ar[:,0], theta_ar[:,1], color = 'red')
    #plt.savefig("contour.png")
    plt.show()

# plot animation of variation of contour with each iteration
def contourAnimation(normalisedData, datasetY, loss_ar, learning_rate):
    fig,ax = plt.subplots()
    x = np.linspace(0, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y, indexing = 'xy')
    z = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(z):
        z[i,j] = getLoss(np.array([[X[i,j]], [Y[i,j]]]), normalisedData, datasetY)

    def animate(i):
        ax.clear()
        ax.contour(X, Y, z, [loss_ar[i]])
        ax.set_xlabel("Theta_0")
        ax.set_ylabel("Theta_1")
        ax.set_title("Variation of contour for error function with iterations, Learning Rate = {0}".format(learning_rate))
    
    contour_ani = animation.FuncAnimation(fig, animate, loss_ar.size-1, interval = 200, blit = False)
    #contour_ani.save('contourAnimation3.mp4', writer = 'ffmpeg', fps = 5)
    plt.show()

# plot the data and hypothesis function
def plotDataHypothesis(theta, datasetX, datasetY, learning_rate):
    fig,ax = plt.subplots()
    x = datasetX[:,1].reshape(datasetX.shape[0],)
    y = datasetY.reshape(datasetY.shape[0],)
    xl, xr = np.min(x), np.max(x)
    yl = theta[0,0]+theta[1,0]*xl
    yr = theta[0,0]+theta[1,0]*xr
    ax.plot([xl,xr], [yl,yr], label = 'Hypothesis Function', color = 'red')
    ax.scatter(x,y)
    ax.legend(loc='best')
    ax.set_title("Data and Learned hypothesis function, Learning Rate = {0}".format(learning_rate))
    ax.set_xlabel("Acidity of wine")
    ax.set_ylabel("Density of wine")
    #plt.savefig("dataAndHypothesis.png")
    plt.show()

#plot loss function vs no of iterations
def plotLossVsIteration(loss_ar, learning_rate):
    fig,ax = plt.subplots()
    x = [i for i in range(loss_ar.size)]
    ax.set_ylabel("Error Value")
    ax.set_xlabel("No. of iterations")
    ax.set_title("Variation of Error Value with iterations, Learning Rate = {0}".format(learning_rate))
    ax.plot(x, loss_ar)
    #plt.savefig("ErrorValue.png")
    plt.show()


# plot animation showing how parameters are changing on the contour line
def contourAnimation2(normalisedData, datasetY, loss_ar, theta_ar):
    fig,ax = plt.subplots()
    x = np.linspace(0, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x,y, indexing = 'xy')
    z = np.zeros((x.size, y.size))
    for (i,j), v in np.ndenumerate(z):
        z[i,j] = getLoss(np.array([[X[i,j]], [Y[i,j]]]), normalisedData, datasetY)

    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_title("Movement of parameters on contour plot")
    i = 0
    while (i<loss_ar.size):
        ax.contour(X,Y,z, [loss_ar[i]], colors = ['gray'], alpha = 0.5)
        i+=1
    
    def animate(i):
        ax.scatter(theta_ar[i,0], theta_ar[i,1], color = 'red')
    
    contour_ani = animation.FuncAnimation(fig, animate, min(loss_ar.size-10, 300), interval = 2, blit = False)
    #contour_ani.save('ThetaMovementOnContour3.mp4', writer = 'ffmpeg', fps = 30)
    plt.show()

#read filename and other parameters from terminal
args = sys.argv[1:]
datasetXfile, datasetYfile, learning_rate = args[0], args[1], float(args[2])
datasetXi = pd.read_csv(datasetXfile, header = None).values
datasetY = pd.read_csv(datasetYfile, header = None).values
#datasetXi = pd.read_csv('data/q1/linearX.csv', header = None).values
#datasetY = pd.read_csv('data/q1/linearY.csv', header = None).values
m,n = datasetXi.shape
mean_ar, std_ar, normalisedData = normalise(datasetXi)
x0 = np.ones((m, 1), dtype= np.float64)
normalisedData = np.concatenate((x0, normalisedData), axis = 1)
theta, theta_ar, loss_ar = linearReg(normalisedData, datasetY, learning_rate)
print(theta)
plotLossVsIteration(loss_ar, learning_rate)
plotDataHypothesis(theta, normalisedData, datasetY, learning_rate)
#contourAnimation(normalisedData, datasetY, loss_ar, learning_rate)
drawMesh(normalisedData, datasetY, theta_ar, loss_ar)
#drawAnimation(normalisedData, datasetY, theta_ar, loss_ar)
#plotContour(normalisedData, datasetY, loss_ar, theta_ar)
#contourAnimation2(normalisedData, datasetY, loss_ar, theta_ar)




