#building a neural network from scratch using python
#the neural network will have 2 hidden layers
#first logistic (hidden) input layer has 4 nodes (logistic regression) (4x785) after the 784 + 1 x1 input layer
#second logistic (hidden) layer 7 nodes (logistic regression) (7x5)
#third logistic (ouput) layer has 10 nodes (logistic regression) (10x8)
#scrap this shit, doesnt actually 
import math
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt    
import random
from numba import njit

def ReLU(Z):
    return np.maximum(Z,0)

def ReLUDeriv(Z):
    return Z > 0

def nn_forwardPropagation(train_X, train_Y, _lambda, theta1, theta2, theta3):
    accuracy = 0
    J = 0
    #forward propagation
    train_X = np.c_[ np.ones((np.shape(train_X)[0],1)), train_X]
    z1 = train_X@np.transpose(theta1)
    a1 = np.c_[ np.ones((np.shape(z1)[0],1)), ReLU(z1)]
    z2 = a1@np.transpose(theta2)
    a2 = np.c_[ np.ones((np.shape(z2)[0],1)), ReLU(z2)]
    #last step
    predValues = np.zeros((10,len(train_Y)))
    y1 = predValues.copy()
    predValues2 = y1.copy()
    for i in range(np.shape(theta3)[0]):
        #one vs all
        predValues[i,:] = sigmoid(a2@np.transpose(theta3[i,:]))
    #in order to calculate accuracy, we just assume that the highest value in our predicted values dictates which value is "right" and look if that's the same as in Y
    #making y1 our "target matrix" where we would like to see a 1 at the place of the right number and 0's everywhere else
    #still trying to optimize this part
    for j in range(len(train_Y)):
        y1[train_Y[j],j] = 1
        predValues2[np.where(predValues[:,j] == max(predValues[:,j])),j] = 1 
        if np.array_equiv(predValues2[:,j], y1[:,j]):
            accuracy += 1/len(train_Y)
        #computing the cost inlc. regularization
        J += np.sum((np.transpose(-y1[:,j])@np.log(predValues[:,j]))  -  (1-np.transpose(y1[:,j]))@np.log(1 - predValues[:,j]))/len(train_Y)
    #calculating the cost with regularization
    J += np.sum(_lambda/(2*len(train_Y)) * theta1[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta2[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta3[:,1:]**2)
    #calculating the derivatives/backprop
    delta3 = predValues - y1
    #print("\n",delta3, "\n",theta3[:,1:])
    delta2 = (np.transpose(delta3) @ theta3[:,1:]) * ReLUDeriv(z2)
    delta1 = delta2 @ theta2[:,1:] * ReLUDeriv(z1)
    D3 = delta3 @ a2 
    D2 = np.transpose(delta2) @ a1
    D1 = np.transpose(delta1) @ train_X
    theta3_grad = D3 / len(train_Y) + np.c_[np.zeros((np.shape(theta3)[0], 1)), (_lambda/len(train_Y))*theta3[:,1:]]
    theta2_grad = D2 / len(train_Y) + np.c_[np.zeros((np.shape(theta2)[0], 1)), (_lambda/len(train_Y))*theta2[:,1:]]
    theta1_grad = D1 / len(train_Y) + np.c_[np.zeros((np.shape(theta1)[0], 1)), (_lambda/len(train_Y))*theta1[:,1:]]
    return J, theta3_grad, theta2_grad, theta1_grad, accuracy
    
def nn_runner(train_X, train_Y, iterations, _lambda, alpha1, alpha2, alpha3, batchsize):
    accuracy_list = []
    epoch_list = []
    hidden1 = int(len(train_Y)/(2 * (785 + 10)))
    hidden2 = hidden1
    
    theta1 = np.random.rand(hidden1,785)
    e = np.sqrt(2) / np.sqrt(np.shape(theta1)[1])
    theta1 = theta1 * 2 * e - e

    theta2 = np.random.rand(hidden2,hidden1 + 1)
    e = np.sqrt(2) / np.sqrt(np.shape(theta2)[1])
    theta2 = theta2 * 2 * e - e

    theta3 = np.random.rand(10,hidden2 + 1)
    e = np.sqrt(2) / np.sqrt(np.shape(theta3)[1])
    theta3 = theta3 * 2 * e - e
    J1 = np.inf
    const1 = 100 / 3
    const2 = const1 * alpha1
    for i in range(1,iterations+1):
        #slowly decreasing the learning rate with every iteration to "guarantee" convergence
        alpha1, alpha2, alpha3 = (const2/(const1 + i)), (const2/(const1 + i)), (const2/(const1 + i))
        averageAccuracy = 0
        averageError = 0
        p = np.random.permutation(len(train_Y))
        batch = train_X[p]
        batchY = train_Y[p]
        for j in range(len(train_Y)//batchsize + 1):
            if len(batchY) > batchsize:
                J, theta3_grad, theta2_grad, theta1_grad, accuracy = nn_forwardPropagation(batch[-batchsize:, :], batchY[-batchsize:], _lambda, theta1, theta2, theta3)
                batch, batchY = batch[:-batchsize], batchY[:-batchsize]
            else:
                J, theta3_grad, theta2_grad, theta1_grad, accuracy = nn_forwardPropagation(batch, batchY, _lambda, theta1, theta2, theta3)
            averageError += J / (len(train_Y)//batchsize + 1)
            averageAccuracy += accuracy / (len(train_Y)//batchsize + 1)
            theta3 -= alpha3*theta3_grad
            theta2 -= alpha2*theta2_grad
            theta1 -= alpha1*theta1_grad

        if i % 1 == 0:
            print("At epoch",i,":", averageError)
            print("avg. Accuracy:",averageAccuracy)
        accuracy_list.append(averageError)
        epoch_list.append(i)
    
    J, theta3_grad, theta2_grad, theta1_grad, accuracy = nn_forwardPropagation(train_X, train_Y, _lambda, theta1, theta2, theta3)
    plt.plot(epoch_list, accuracy_list)
    plt.show()
    print("This amounts to a total accuracy on training data of", accuracy, "with an error of", J)
    return theta3, theta2, theta1
        
def sigmoid(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    train_X = (train_X - np.average(train_X)) / np.std(train_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
    test_X = (test_X - np.average(train_X)) / np.std(train_X)
    cost = []
    hyperparameters = []
    g = 0
    _lambda = 7
    iterations = 20
    alpha1 = 0.21
    alpha2 = alpha1
    alpha3 = alpha1
    batchsize = 1024
    theta3, theta2, theta1 = nn_runner(train_X, train_y, iterations, _lambda, alpha1, alpha2, alpha3, batchsize)
    J, theta3_grad, theta2_grad, theta1_grad, accuracy = nn_forwardPropagation(test_X, test_y, _lambda, theta1, theta2, theta3)
    print("And an accuracy on test of", accuracy, "with an error of", J)
    
if __name__ == '__main__':
    main()
