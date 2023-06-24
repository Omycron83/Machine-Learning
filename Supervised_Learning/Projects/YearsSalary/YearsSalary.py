import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def step_gradient(b, m, Y, X, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = len(Y)
    for i in range(N):
        x = X[i]
        y = Y[i]
        y1 = m*x + b
        b_gradient += (-1/N) * (y-y1)
        m_gradient += (-1/N) * x * (y-y1)

    new_b = b - (learningRate*b_gradient)
    new_m = m - (learningRate*m_gradient)
    return(new_b, new_m)

def callGradient(b, m, Y, X, learningRate, epochs):
    b1 = b
    m1 = m
    for i in range(epochs):
        b1, m1 = step_gradient(b1, m1, Y, X, learningRate)

    return(b1, m1)

def error(b, m, Y, X):
    totalError = 0
    error = 0

    for i in range(len(Y)):
        x = X[i]
        y = Y[i]
        error = (y - (m*x + b))**2
        totalError += error / (len(Y)*2)
        print ("At Row {0}, using b = {1} and m = {2}, Error = {3}".format(i, b, m, error))

    print('\n The total error is:',totalError)

def main():
    data1 = pd.read_csv('Salary_Data.csv')
    Y = data1['Salary'].tolist()
    X = data1['YearsExperience'].tolist()
    plt.xlim = (0, max(X))
    plt.ylim = (0, max(Y))
    plt.plot(X, Y, 'o', color = 'black')
    plt.show()
    learningRate = 0.01
    initial_b = 1
    initial_m = 1
    epochs = 1000
    print ("\n First compute Error for each row by using equation y_predicted = mx +b and error =  (y - y_predicted) ^2 / len(points) by using random b = {0}, and m = {1} \n".format(initial_b, initial_m))
    error(initial_b, initial_m, Y, X)
    print ("\n Now, let's run gradient_descent_runner to get new m and b with learning rate of {1} and {0} iterations \n".format(epochs, learningRate))
    (b, m) = callGradient(initial_b, initial_m, Y, X, learningRate, epochs)
    print(b,m)
    error(b, m, Y, X)
    print ("\n After {0}nd iterations final b = {1}, m = {2} \n".format(epochs,b,m))
    X_test = 4
    y_test = m* X_test + b
    print ("\n The Salary should be {0} \n".format(y_test))
    x = np.linspace(min(X), max(X), 50)
    y = m*x + b
    plt.plot(x,y, '-r')
    plt.plot(X, Y, 'o', color = 'black')
    plt.show()

if __name__ == '__main__':
    main()
