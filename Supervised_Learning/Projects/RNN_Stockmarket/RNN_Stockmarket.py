from pandas_datareader import data as pdr
import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt    
import random
#this algorithm aims to implement a many-to-one algorithm, where we predict a stockprise after having
#seen the stockprices after a certain number of days, kind of emulating "having a feel for prices"
#we do this manually, as usual, because we actually do love this kind of bullshit as we learn more
#also, we will implementa many to many solution, but not actually probably use it, as we do need to do stuff
#this is the first iterations as i chose to refactor some stuff down the line to make backprop easier


def ReLU(Z):
    return (Z > 0) * Z

def ReLUDeriv(Z):
    return (Z > 0)


def propagationManyToOne(x, theta_neur, theta_out, neurons, bias, y):
    return 0

def propagationManyToMany(x, theta_neur, theta_out, neurons, bias, y):
    #forward propagation:
    cache = np.zeros((1, x.shape[0], neurons))
    no_ReLU = cache.copy
    for i in range(x.shape[1]- 1):
        cache = np.vstack([cache, ReLU(np.hstack((cache[i,:,:], x[:, i].reshape(x.shape[0], 1))) @ theta_neur).reshape(1, x.shape[0], neurons)])
        no_ReLU = np.vstack([cache, (np.hstack((cache[i,:,:], x[:, i].reshape(x.shape[0], 1))) @ theta_neur).reshape(1, x.shape[0], neurons)])

    y_hat = np.concatenate((np.ones((cache.shape[0], cache.shape[1], 1)), cache), axis = 2) @ theta_out
    J = ((y_hat.reshape(y.shape) - y)**2).sum() / (2 * y_hat.size)
    #backwards propagation (as all losses add up and are averaged, we average the derivatives?)
    lossDeriv = y_hat.reshape(y.shape) - y
    theta_out_deriv = np.tensordot(lossDeriv, np.concatenate((np.ones((cache.shape[0], cache.shape[1], 1)), cache), axis = 2), axes=([0, 1], [1, 0])).reshape(theta_out.shape) / cache.size
    theta_neur_deriv = lossDeriv[:, -1].reshape([lossDeriv.shape[0], 1]) @ theta_out.T * ReLUDeriv(np.hstack((cache[-1,:,:], x[:, -1].reshape(x.shape[0], 1))))
    theta_neur_deriv = theta_neur_deriv.T @ cache[-2, :, :] / (2 * y_hat.size)
    print(theta_neur_deriv.shape, theta_neur.shape)
    #print(lossDeriv.shape, theta_out.shape, theta_out_deriv.shape)
    #theta_neur_deriv = lossDeriv @ theta_out  

    
    return J, theta_out_deriv, y_hat, theta_neur_deriv
    
def rnn_runner(_lambda, alpha, batchsize, iterations, x, y):
    neurons = 100
    
    theta_neur = np.random.rand(neurons + 1, neurons)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta_neur)[1]))
    theta_neur = theta_neur * 2 * e - e

    theta_out = np.random.rand(neurons + 1, 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta_out)[1]))
    theta_out = theta_out * 2 * e - e

    bias = np.random.rand(2, 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(bias)[1]))
    bias = bias * 2 * e - e

    for i in range(iterations):
        J, theta_out_deriv, y_hat, theta_neur_deriv = propagationManyToMany(x, theta_neur, theta_out, neurons, bias, y)
        theta_out -= alpha * theta_out_deriv
        theta_neur -= alpha * theta_neur_deriv
        #print("J:", J)
        print(y[-1, 3], y_hat[-1, 3])
        
def main():
    years = 5
    data = (pdr.get_data_yahoo("SPY", (dt.datetime.now() - dt.timedelta(days=years * 365)), dt.datetime.now()))["Adj Close"].to_numpy()
    _lambda = 0
    alpha = 0.00001
    batchsize = 40
    amountOfDays = 10
    iterations = 300
    x = np.empty((0 , amountOfDays))
    for i in range(data.shape[0] - amountOfDays):
        x = np.vstack((x, data[i: i + amountOfDays]))   
    y = x[1:, :]
    x = x[:-1, :]
    
    
    rnn_runner(_lambda, alpha, batchsize, iterations, x, y)
    
if __name__ == '__main__':
    main()
