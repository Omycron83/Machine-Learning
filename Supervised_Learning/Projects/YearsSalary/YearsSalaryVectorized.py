import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def optimalSolution(X, Y):
    return ((X.T@X)**-1) * X.T @ Y

def gradientDescent(theta, Y, X, learningRate, iterations):
    for i in range(iterations):
        theta = [theta[i] - learningRate/len(Y) * np.sum(np.subtract((np.dot(X,theta)),Y) * np.transpose(X[:,i])) for i in range(len(theta))]
    return theta
    
def error(theta, Y, X):
    error = np.sum(np.subtract(np.dot(X,np.transpose(theta)),Y)**2)/(len(Y)*2)
    return error

def main():
    data1 = pd.read_csv('Salary_Data.csv')
    Y = data1['Salary'].tolist()
    X = data1['YearsExperience'].tolist()
    X = np.stack((np.ones((len(X),)), np.array(X)), axis=-1)
    X = np.array(X)
    Y = np.array(Y)
    learningRate = 0.01
    theta = np.ones((2,))
    iterations = 10000
    print("\n The error using default values for theta of {0} results in an error of {1}\n".format(theta,error(theta, Y, X)))
    theta = gradientDescent(theta, Y, X, learningRate, iterations)
    print("\n After running gradient descent {0} times with a learning rate of {1}, we get values of {2}".format(iterations, learningRate,theta))
    print("\n This results in an error of {0}".format(error(theta, Y, X)))
    X_test = 10
    y_test = round(theta[1]* X_test + theta[0],2)
    print ("\n This means the salary with for example {0} years of experience should be {1} \n".format(X_test, y_test))
    x = np.linspace(min(X[:,1]), max(X[:,1]), 2)
    y = theta[1]*x + theta[0]
    plt.plot(x,y, '-r')
    plt.plot(X[:,1], Y, 'o', color = 'black')
    plt.show()
    print("The optimal solution is", optimalSolution(x, y))

if __name__ == '__main__':
    main()
