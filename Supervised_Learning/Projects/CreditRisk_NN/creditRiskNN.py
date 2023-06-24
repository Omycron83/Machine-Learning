#The object of the python code here is building a Neural Network that can (hopefully) correctly predict if a customer will default on their loan
#not sure about dimensions yet, but lets just start and first import some basic wonderful libraries as i dont intend to write code defining matrix operations or reading in data myself
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#first lets get the data and prepare it:
data = pd.read_csv('credit_risk_dataset.csv').dropna()
#initializing our research data np matrix with the bias column
research_data = [np.ones(len(data.index))]
#first the different features we got
#we convert all columns to strings with their respective names
for col in data.columns:
    globals()['%s' % col] = data[col].to_numpy()
#now we need to convert all categories (for example in person_home_ownership where there is rent, own etc.) to numbers corresponding to them in order to
#be able to work with them
    if globals()['%s' % col].dtype == 'object':
        categories = np.unique(globals()['%s' % col])
        number = -1
        for i in categories:
            number += 1
            globals()['%s' % col] = [int(number) if x == i else x for x in globals()['%s' % col]]
    globals()['%s' % col] = np.array(globals()['%s' % col]).astype('float64')
    research_data = np.vstack([research_data, globals()['%s' % col]])
#as we now have a bunch of column vectors and we want our data to be in rows, lets take the transpose of our matrix
research_data = research_data.T
#now we gotta split that into our features and our prediction values (i.e. if the person defaults on this loan or not)
#we figure out the column our relevant feature is in (this is loan_status here), and then splice that out while adding together the remaining splices around it column-wise
X = np.concatenate((research_data[:, :9], research_data[:, 10:]), 1)
Y = research_data[:, 9]
#now, lets normalize the data:
for i in range(1, len(X[0, :].T)):
    X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
#now it turns out that we have a skewed dataset (around 80% of the people dont default), which lead the algorithm to, in the first tries, just predict everything to NOT default
#as that obviously is a shitty way to actually predict if someone will default (yeah, he probably wont so give that guy with 0 income and a past default who uses this to repay other debt the loan)
#we will now just conveniently cut out a bunch of non-default examples and train with that, testing with the all-including dataset afterwards (we will first shuffle the data to avoid any problems with potential data)
p = np.random.permutation(len(Y))
X = X[p]
X_real = np.copy(X)
Y = Y[p]
Y_real = np.copy(Y)
#then see how many examples we will need to cut out in order to get to 50/50 default/no default
n = - (len(Y) - 2 * np.sum(Y))
#then, as long as we havent deleted enough rows yet we figure out if a row is non-default, and if it is add it to a list of the should-be deleted indeces (as doing it right there would have other problems i have no idea and dont want to get into fixing rn as they would require me to research a bunch of stuff)
delets = []
for i in range(len(Y)):
    if n < 0:
        if Y[i] == 0:
            n += 1
            delets.append(i)
    else:
        break
Y = np.delete(Y, delets, axis = 0)
X = np.delete(X, delets, axis = 0)
print(np.sum(Y) / len(Y))
#now lets get on to the Neural Network, shall we?
#(We will try to use an implementation that is vectorized as far as possible in order to insure we have small calculation times)
#We will use ReLU in our layers for that reason as well and only in our output layer we will use sigmoid. Ill probably use a 3 layer-network for now and change it if it we require more/less complexity

# first lets define some functions
def ReLU(Z):
    return np.maximum(Z,0)

def ReLUDeriv(Z):
    return Z > 0

def sigmoid(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def nn_forwardPropagation(train_X, train_Y, _lambda, theta1, theta2, theta3):
    train_Y = train_Y.reshape(np.shape(train_Y)[0], 1)
    #forward propagation
    z1 = train_X@np.transpose(theta1)
    a1 = np.c_[ np.ones((np.shape(z1)[0],1)), ReLU(z1)]
    z2 = a1@np.transpose(theta2)
    a2 = np.c_[ np.ones((np.shape(z2)[0],1)), ReLU(z2)]
    #last step
    predValues = sigmoid(a2@np.transpose(theta3))
    #print(predValues)
    J = np.sum(-train_Y*np.log(predValues)  -  (1-train_Y)*np.log(1 - predValues)) /len(train_Y)
    #in order to calculate accuracy, we just assume that the highest value in our predicted values dictates which value is "right" and look if that's the same as in Y
    #we will use the F1 score instead of accuracy (still optional for anybody to try out, the formula being down there), as we can figure out the performance on skewed data using this
    tp = np.sum((np.around(predValues, 0) == 1) & (train_Y == 1))
    fp = np.sum((np.around(predValues, 0) == 1) & (train_Y == 0))
    fn = np.sum((np.around(predValues, 0) == 0) & (train_Y == 1))
    accuracy = tp / (tp + 1/2 * (fp + fn))
    #accuracy = np.sum(np.around(predValues, 0) == train_Y) / len(train_Y)
    #calculating the cost with regularization
    J += np.sum(_lambda/(2*len(train_Y)) * theta1[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta2[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta3[1:]**2)
    #calculating the derivatives/backprop
    delta3 = predValues - train_Y
    #print("\n",delta3, "\n",theta3[:,1:])
    delta2 = delta3 @ theta3[:,1:] * ReLUDeriv(z2)
    delta1 = delta2 @ theta2[:,1:] * ReLUDeriv(z1)
    D3 = delta3.T @ a2 
    D2 = np.transpose(delta2) @ a1
    D1 = np.transpose(delta1) @ train_X
    theta3_grad = D3 / len(train_Y) + np.c_[np.zeros((np.shape(theta3)[0], 1)), (_lambda/len(train_Y))*theta3[:,1:]]
    theta2_grad = D2 / len(train_Y) + np.c_[np.zeros((np.shape(theta2)[0], 1)), (_lambda/len(train_Y))*theta2[:,1:]]
    theta1_grad = D1 / len(train_Y) + np.c_[np.zeros((np.shape(theta1)[0], 1)), (_lambda/len(train_Y))*theta1[:,1:]]
    print(np.sum(np.round(predValues, 0)) / len(predValues))
    return J, theta3_grad, theta2_grad, theta1_grad, accuracy
    
def nn_runner(train_X, train_Y, iterations, _lambda, alpha, batchsize):
    accuracy_list = []
    epoch_list = []
    hidden1 = int(len(train_Y)/(2 * (np.shape(train_X)[1] + 1)))
    hidden2 = hidden1
    
    theta1 = np.random.rand(hidden1,np.shape(train_X)[1])
    e = np.sqrt(2) / (np.sqrt(np.shape(theta1)[1]))
    theta1 = theta1 * 2 * e - e

    theta2 = np.random.rand(hidden2,hidden1 + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta2)[1]))
    theta2 = theta2 * 2 * e - e

    theta3 = np.random.rand(1,hidden2 + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta3)[1]))
    theta3 = theta3 * 2 * e - e
    
    const1 = 100 / 20
    const2 = const1 * alpha
    for i in range(1,iterations+1):
        #slowly decreasing the learning rate with every iteration to "guarantee" convergence
        alpha = (const2/(const1 + i))
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
            theta3 -= alpha*theta3_grad
            theta2 -= alpha*theta2_grad
            theta1 -= alpha*theta1_grad

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
        
batchsize = 4096
_lambda = 0
iterations = 50
alpha = 0.21
theta3, theta2, theta1 = nn_runner(X, Y, iterations, _lambda, alpha, batchsize)
J, theta3_grad, theta2_grad, theta1_grad, accuracy = nn_forwardPropagation(X_real, Y_real, _lambda, theta1, theta2, theta3)
print("And an accuracy on test of", accuracy, "with an error of", J)
