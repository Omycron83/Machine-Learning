#titanic logistic regression cuz im bored ig
import numpy as np
import pandas as pd
import math
import random
from decimal import Decimal
np.seterr(divide = 'ignore')

def pred(theta, X, y):
    accuracy = 0
    for i in range(len(y)):
        if sigmoid(theta@X[i]) >= 0.5:
            predValue = 1
        else:
            predValue = 0
        if predValue == y[i]:
            accuracy += 1/len(y)
    return accuracy

def SigmoidGradient(theta, X, y, _lamda):
    grad = np.ones(theta.size)
    grad[0] = np.sum((sigmoid(X@theta) - y)*X[:,0])/len(y)
    for i in range(1,len(grad)):
        grad[i] = np.sum((sigmoid(X@theta) - y)*X[:,i])/len(y) + _lambda * theta[i]/len(y)
    return grad

def gradientRunner(theta, X, Y, _lambda, learningRate, iterations):
    for i in range(iterations):
        theta -= learningRate*SigmoidGradient(theta, X, Y, _lambda)
        if i % 100 == 0:
            print("At iteration {0} cost {1}".format(i, logisticCost(theta, X, Y, _lambda)))
    return theta

def sigmoid(X):
    sig = 1 / (1 + np.exp(-X))
    sig = np.minimum(sig, 0.9999)
    sig = np.maximum(sig, 0.0001)
    return sig

def logisticCost(theta, X, Y, _lambda):
    m = len(Y)
    J = ((-np.transpose(Y)@np.log(sigmoid(X@Theta)))  -  (1-np.transpose(Y))@np.log(1 - sigmoid(X@theta)))/m + _lambda/(2*m) * np.sum(np.power(Theta[2:len(theta)],2))
    return J

data = pd.read_csv('tested.csv').dropna()
Y = data["Survived"].tolist()
Class = data["Pclass"].tolist()
Sex = data["Sex"].tolist()
Age = data["Age"].tolist()
SibSp = data["Age"].tolist()
Parch = data["Age"].tolist()
Fare = data["Age"].tolist()
bias = np.ones(np.size(Fare))
for i in range(len(Sex)):
    if Sex[i] == "female":
        Sex[i] = 0
    if Sex[i] == "male":
        Sex[i] = 1
X = np.stack((bias,Class, Sex, Age, SibSp, Parch, Fare), axis=-1)
f = X.shape[0]
X, X_test = X[f // 3:,:],X[:f// 3,:]
Y, Y_test = Y[f // 3:],Y[:f // 3]
learningRate = 0.001
iterations = 100000
_lambda = 10
Theta = random.sample(range(0, 40), len(X[1]))
Theta = np.divide(Theta, 10)
print("Cost at initial guesses of", Theta, "and lamda =", _lambda, ":",logisticCost(Theta, X, Y, _lambda))
theta = gradientRunner(Theta, X, Y, _lambda, learningRate, iterations)
print("\n Our final values for theta are:", theta, "with an error of", logisticCost(theta, X, Y, _lambda))
print("Using this, lets now figure out the error on our training dataset:")
print("The accuracy of the model is at:", int((pred(theta, X, Y)+0.001)*100),"%")
print("The accuracy on test data is then:", int((pred(theta, X_test, Y_test)+0.001)*100),"%, with a cost of", logisticCost(theta, X_test, Y_test, 0))
