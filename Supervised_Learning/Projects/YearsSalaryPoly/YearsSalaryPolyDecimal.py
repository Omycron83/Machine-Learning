import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decimal import *
getcontext().prec = 20

def gradientDescent(theta, X, Y, learningRate, iterations):
    for i in range(iterations):
        theta = [theta[i] - Decimal(learningRate/len(Y)) * np.sum(np.subtract((np.dot(X,theta)),Y)* np.transpose(np.array(X)[:,i])) for i in range(len(theta))]
    return theta


def error(values, data, Y):
    totalError = Decimal(0)
    for i in range(len(Y)):
        error = Decimal((Y[i] - (np.dot(values, data[i])))**2)
        totalError += error / (2*len(Y))
    print('\n The total error is:',totalError)

def normalEquation(X, Y):
    X = [[float(i) for i in sublist] for sublist in X]
    Y = [float(i) for i in Y]
    beta = np.dot((np.linalg.inv(np.dot(np.transpose(X),X))), np.dot(np.transpose(X),Y))
    return [Decimal(i) for i in beta]

def main():
    data1 = pd.read_csv('Salary_Data.csv')
    Y = data1['Salary'].tolist()
    X = data1['YearsExperience'].tolist()
    
    learningRate = 0.0000000005
    epochs = 10
    degree = 11
    x = [0] * degree
    for i in range(degree):
       x[i] = np.array(X)**i
    data = np.stack((i for i in x), axis=-1)
    values = [Decimal(1)] * len(data[0])
    data = [[Decimal(i) for i in sublist] for sublist in data]
    Y = [Decimal(i) for i in Y]
    print(X)
    
    error(values, data, Y)
    print ("\n Now, let's run gradient_descent_runner to get new m and b with learning rate of {1} and {0} iterations \n".format(epochs, learningRate))
    values = gradientDescent(values, data, Y, learningRate, epochs)
    print(values)
    error(values, data, Y)
    print("The calculated values using the normal equation are:")
    #testing it against the normal equation
    values2 = normalEquation(data, Y)
    print(values2)
    error(values2, data, Y)
    #visualizing the results
    X_test = 10
    y_test = Decimal(np.dot(data[i], values2))
    print ("\n The Salary should be {0} \n".format(y_test))
    x_axis = np.linspace(float(min(np.array(data)[:,1])),float(max(np.array(data)[:,1])),500)
    x_axis = [Decimal(i) for i in x_axis]
    y_axis = [[np.sum(np.dot(values2,[x_axis[j]**i for i in range(len(values2))]))] for j in range(len(x_axis))]
    plt.plot(np.array(data)[:,1], Y, 'o', color = 'black')
    plt.plot(x_axis, y_axis, '-r')
    plt.show()


if __name__ == '__main__':
    main()
