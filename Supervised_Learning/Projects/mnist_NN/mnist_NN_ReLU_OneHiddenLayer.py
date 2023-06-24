#mnit ReLU but we only have one hidden layer to avoid overfitting:
import math
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt    
import random

def ReLU(Z):
    return np.maximum(Z,0)

def ReLUDeriv(Z):
    return Z > 0

def nn_forwardPropagation(train_X, train_Y, _lambda, theta1, theta3):
    accuracy = 0
    J = 0
    #forward propagation
    train_X = np.c_[ np.ones((np.shape(train_X)[0],1)), train_X]
    z1 = train_X@np.transpose(theta1)
    a1 = np.c_[ np.ones((np.shape(z1)[0],1)), ReLU(z1)]
    #last step
    predValues = np.zeros((10,len(train_Y)))
    y1 = predValues.copy()
    predValues2 = y1.copy()
    for j in range(len(train_Y)):
        for i in range(np.shape(theta3)[0]):
            #one vs all
            predValues[i,j] = sigmoid(a1[j,:]@np.transpose(theta3[i,:]))
        #making y1 our "target matrix" where we would like to see a 1 at the place of the right number and 0's everywhere else 
        y1[train_Y[j],j] = 1
        #J += np.sum((predValues[:,j] - y1[:,j])**2) / len(train_Y)
        J += np.sum((np.transpose(-y1[:,j])@np.log(predValues[:,j]))  -  (1-np.transpose(y1[:,j]))@np.log(1 - predValues[:,j]))/len(train_Y)
        #in order to calculate accuracy, we just assume that the highest value in our predicted values dictates which value is "right" and look if that's the same as in Y
        predValues2[np.where(predValues[:,j] == max(predValues[:,j])),j] = 1 
        if np.array_equiv(predValues2[:,j], y1[:,j]):
            accuracy += 1/len(train_Y)
    #calculating the cost with regularization
    J += np.sum(_lambda/(2*len(train_Y)) * theta1[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta3[:,1:]**2)
    #calculating the derivatives/backprop
    delta3 = predValues - y1
    #print("\n",delta3, "\n",theta3[:,1:])
    delta2 = (np.transpose(delta3) @ theta3[:,1:]) * ReLUDeriv(z1)
    D3 = delta3 @ a1
    D1 = np.transpose(delta2) @ train_X
    theta3_grad = D3 / len(train_Y) + np.c_[np.zeros((np.shape(theta3)[0], 1)), (_lambda/len(train_Y))*np.absolute(theta3[:,1:])]
    theta1_grad = D1 / len(train_Y) + np.c_[np.zeros((np.shape(theta1)[0], 1)), (_lambda/len(train_Y))*np.absolute(theta1[:,1:])]
    return J, theta3_grad, theta1_grad, accuracy
    
def nn_runner(train_X, train_Y, iterations, _lambda, alpha1, alpha2, alpha3, batchsize):
    
    hidden1 = int(len(train_Y)/(2 * (785 + 10)))
    
    theta1 = np.random.rand(hidden1,785)
    e = np.sqrt(2) / np.sqrt(np.shape(theta1)[1])
    theta1 = theta1 * 2 * e - e

    theta3 = np.random.rand(10,hidden1 + 1)
    e = np.sqrt(2) / np.sqrt(np.shape(theta3)[1])
    theta3 = theta3 * 2 * e - e
    
    for i in range(1,iterations+1):
        averageAccuracy = 0
        averageError = 0
        p = np.random.permutation(len(train_Y))
        batch = train_X[p]
        batchY = train_Y[p]
        for j in range(len(train_Y)//batchsize + 1):
            if len(batchY) > batchsize:
                J, theta3_grad, theta1_grad, accuracy = nn_forwardPropagation(batch[-batchsize:, :], batchY[-batchsize:], _lambda, theta1, theta3)
                batch, batchY = batch[:-batchsize], batchY[:-batchsize]
            else:
                J, theta3_grad, theta1_grad, accuracy = nn_forwardPropagation(batch, batchY, _lambda, theta1, theta3)
            averageError += J / (len(train_Y)//batchsize + 1)
            averageAccuracy += accuracy / (len(train_Y)//batchsize + 1)
            theta3 -= alpha3*theta3_grad
            theta1 -= alpha1*theta1_grad
        if i % 1 == 0:
            print("At epoch",i,":", averageError)
            print("avg. Accuracy:",averageAccuracy)
    J, theta3_grad, theta1_grad, accuracy = nn_forwardPropagation(train_X, train_Y, _lambda, theta1, theta3)
    print("This amounts to a total accuracy on training data of", accuracy, "with an error of", J)
    return theta3, theta1
        
def sigmoid(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    plt.imshow(train_X[0], cmap='gray')
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    train_X = (train_X - np.average(train_X)) / np.std(train_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
    test_X = (test_X - np.average(train_X)) / np.std(train_X)
    _lambda = 0.3
    iterations = 120
    alpha1 = 0.25
    alpha2 = 0.25
    alpha3 = 0.25
    batchsize = 1000
    theta3, theta1 = nn_runner(train_X, train_y, iterations, _lambda, alpha1, alpha2, alpha3, batchsize)
    
    J, theta3_grad, theta1_grad, accuracy = nn_forwardPropagation(test_X, test_y, _lambda, theta1, theta3)
    print("This amounts to a total accuracy on test data of", accuracy, "with an error of", J)
    #print(theta1, theta2, theta3)
    
if __name__ == '__main__':
    main()
