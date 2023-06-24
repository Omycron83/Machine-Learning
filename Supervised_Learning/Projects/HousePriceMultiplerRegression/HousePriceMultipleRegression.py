#linear regression algorithm that predicts house prices
import pandas as pd
import numpy as np

def gradient_descent(values,data, y, learningRate):
    N = len(data)
    gradient = [0]*len(values)

    for i in range(len(y)):
        yReal = np.dot(data[i],values)
        for j in range(len(values)):
            gradient[j] += (-1/N) * (y[i] - yReal) * data[i][j]

    for i in range(len(values)):
        values[i] -= learningRate * gradient[i]
        
    return(values)

def gradient_descent_runner(values, data, y, learningRate ,iterations):
    for i in range(iterations):
        values = gradient_descent(values, data, y, learningRate)
    return(values)

def computeErrors(values, data, y):
    totalError = 0
    for i in range(0, len(y)):
        error = (y[i] - (np.dot(values, data[i])))**2
        totalError += error / len(y)
    return totalError

def normalEquation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
    return beta

def main():
    data = pd.read_csv('USA_Housing.csv')
    y = data['Price'].tolist()
    b_ = [1] * len(y)
    x1 = data['Avg. Area Income'].tolist()
    x2 = data['Avg. Area House Age'].tolist()
    x3 = data['Avg. Area Number of Rooms'].tolist()
    x4 = data['Avg. Area Number of Bedrooms'].tolist()
    x5 = data['Area Population'].tolist()

    
    x1 = (np.array(x1) - np.average(x1))/np.ptp(x1)
    x2 = (np.array(x2) - np.average(x2))/np.ptp(x2)
    x3 = (np.array(x3) - np.average(x3))/np.ptp(x3)
    x4 = (np.array(x4) - np.average(x4))/np.ptp(x4)
    x5 = (np.array(x5) - np.average(x5))/np.ptp(x5)
    y = (np.array(y) - np.average(y))/np.ptp(y)

    learningRate = 0.6
    iterations = 500

    data = np.stack((b_, x1, x2, x3, x4, x5), axis=-1)
    values = [1] * len(data[0])

    print("\n First compute Error using the default assumptions thetas \n".format(values))
    print(computeErrors(values, data, y))
    print("\n Now lets run the gradient descent algorithm with a learning rate of {0} and {1} epochs \n".format(learningRate, iterations))
    values = gradient_descent_runner(values, data, y, learningRate, iterations)
    print("\n This got us the values {0} \n".format(values))
    error = computeErrors(values, data, y)
    print("\nThis gives us a total error of", error)

    print("\nLets compare the result to the solution of the normal equation:")
    values1 = normalEquation(data, y)
    print(values1)
    error = computeErrors(values1, data, y)
    print("\n The error using that is:", error)
    
if __name__ == '__main__':
    main()
