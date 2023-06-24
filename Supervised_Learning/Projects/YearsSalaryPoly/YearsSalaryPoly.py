import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def step_gradient(values, data, Y, learningRate):
    gradient = [0] * len(values)
    for i in range(len(Y)):
        yReal = np.dot(data[i], values)
        for j in range(len(values)):
            gradient[j] += (-1/len(data)) * (Y[i] - yReal) * data[i][j]

    for i in range(len(values)):
        values[i] -= learningRate * gradient[i]
                            
    return(values)

def callGradient(values, data, Y, learningRate, epochs):
    for i in range(epochs):
        values = step_gradient(values, data, Y, learningRate)

    return(values)

def error(values, data, Y):
    totalError = 0
    for i in range(len(Y)):
        error = (Y[i] - (np.dot(values, data[i])))**2
        totalError += error / len(Y)
    print('\n The total error is:',totalError)

def normalEquation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
    return beta

def main():
    data1 = pd.read_csv('Salary_Data.csv')
    Y = data1['Salary'].tolist()
    X = data1['YearsExperience'].tolist()

    learningRate = 0.0003
    epochs = 10000
    degree = 3
    x = [0] * degree
    
    for i in range(degree):
       x[i] = np.array(X)**i
    
    data = np.stack((i for i in x), axis=-1)
    values = [1] * len(data[0])

    error(values, data, Y)
    print ("\n Now, let's run gradient_descent_runner to get new m and b with learning rate of {1} and {0} iterations \n".format(epochs, learningRate))
    values = callGradient(values, data, Y, learningRate, epochs)
    print(values)
    error(values, data, Y)
    print("The calculated values using the normal equation are:")
    values2 = normalEquation(data, Y)
    print(values2)
    error(values2, data, Y)

    X_test = 4
    y_test = 0
    for i in range(0, len(values)):
        y_test += (X_test**i) * values[i]
        print(values[i], X_test**i)
    print ("\n The Salary should be {0} \n".format(y_test))

if __name__ == '__main__':
    main()
