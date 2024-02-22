#implementing a standard NN in a "nice", object oriented way to kinda learn that or smth
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numba
from skimage.util.shape import view_as_windows

#We generally define a neural network
class neural_network(ABC):
    @abstractmethod
    def forward_propagation(self):
        raise NotImplementedError
    @abstractmethod
    def backward_propagation(self):
        raise NotImplementedError


#Current project, semi in construction
class modular_NN(neural_network):
    def __init__(self, nodes, non_linear, non_linear_derivative, output_func, output_func_deriv, error, error_deriv):
        self.nodes = nodes
        nodes[0] -= 1
        self.non_linear = non_linear
        self.non_linear_derivative = non_linear_derivative
        self.output_func = output_func
        self.output_func_deriv = output_func_deriv
        self.error = error
        self.error_deriv = error_deriv
        self.layers = []
        for i in range(len(self.nodes) - 2):
            self.layers.append(self.feedforward_layer(nodes[i] + 1, nodes[i + 1], non_linear, non_linear_derivative))
        self.layers.append(self.output_layer(nodes[i + 1] + 1, nodes[i + 2], output_func, output_func_deriv, error, error_deriv))

    class feedforward_layer:
        def __init__(self, input_size, output_size, non_linear, non_linear_derivative):
            #Initializing layer weight
            self.theta = np.random.rand(input_size, output_size)
            #One may argue for/against using He-Initialization in favor of Xavier initialization etc.
            e = np.sqrt(2) / np.sqrt(self.theta.shape[0])
            self.theta = self.theta * 2 * e - e
            #The functions
            self.non_linear = non_linear
            self.non_linear_derivative = non_linear_derivative
            #The direct attributes
            self.z = None
            self.a = None
            #Gradient parameter
            self.derivative = np.zeros((self.theta.shape[0], self.theta.shape[1]))

        @numba.jit(forceobj = True)
        def forprop(self, prevLayer, dropout = 0):
            self.prevLayer = prevLayer
            self.z = prevLayer @ self.theta
            if dropout > 0:
                layer_dropout = np.random.rand(self.z.shape[0], self.z.shape[1])
                self.z *= layer_dropout > (1-dropout) / (1-dropout)
            self.a = np.hstack((np.ones((self.z.shape[0], 1)), self.non_linear(self.z)))
            return self.a
        
        @numba.jit(forceobj = True) 
        def backprop(self, delta_prev, weights_prev):
            self.delta_curr = delta_prev @ weights_prev.T * self.non_linear_derivative(self.z)
            self.derivative = self.prevLayer.T @ self.delta_curr
            return self.delta_curr, self.theta[1:, :]

    class output_layer(feedforward_layer):
        def __init__(self, input_size, output_size, output_func, output_func_deriv, error, error_deriv):
            super().__init__(input_size, output_size, output_func, output_func_deriv)
            self.error = error
            self.error_deriv = error_deriv

        @numba.jit(forceobj = True)
        def forprop(self, prevLayer, pred):
            self.prevLayer = prevLayer
            self.z = prevLayer @ self.theta
            self.output = self.non_linear(self.z)
            self.pred = pred
            return self.error(self.output, self.pred)
        
        @numba.jit(forceobj = True)
        def backprop(self):
            self.cost_deriv = self.error_deriv(self.output, self.pred)
            self.delta_output = self.non_linear_derivative(self.output) * self.cost_deriv
            self.derivative = self.prevLayer.T @ self.delta_output
            return self.delta_output, self.theta[1:, :]
    
    @numba.jit(forceobj = True)
    def forward_propagation(self, inputs, pred, dropout = []):
        if dropout == []:
            dropout = [0 for i in range(len(self.layers))]
        layer_output = inputs
        for i in range(len(self.layers) - 1):
            layer_output = self.layers[i].forprop(layer_output, dropout[i])
        return self.layers[i + 1].forprop(layer_output, pred)

    @numba.jit(forceobj = True)
    def backward_propagation(self):
        delta_prev, weights_prev = self.layers[-1].backprop()
        for i in range(1, len(self.layers)):
             delta_prev, weights_prev = self.layers[::-1][i].backprop(delta_prev, weights_prev)
    

    class conv_layer:
        @numba.jit(forceobj = True)
        def padding(self, pic, padX, padY):
            return np.pad(pic, (padX, padY), constant_values=(0, 0))

        @numba.jit(forceobj = True)
        def avg_pooling(self, input):
            return np.average(input)

        @numba.jit(forceobj = True)
        def max_pooling(self, input):
            return np.max(input)

        #Vectorized + striding window
        @numba.jit(forceobj = True)
        def memory_strided_im2col(self, pic, filter, padX, padY, stride):
            pic = self.padding(pic, padX, padY)
            im2col_matrix = view_as_windows(pic, filter.shape, step = stride).reshape(filter.size, int((pic.shape[0] - filter.shape[0]) / stride + 1) * int((pic.shape[1] - filter.shape[1]) / stride + 1))
            return (filter.flatten() @ im2col_matrix).reshape(int((pic.shape[0] + 2*padX - filter.shape[0]) / stride + 1), int((pic.shape[1] + 2*padY - filter.shape[1]) / stride + 1)) 
        

        def __init__(self) -> None:
            pass
    
    class pooling_layer:
        def __init__(self) -> None:
            pass
    
    #Optimization methods
    @numba.jit(forceobj = True)
    def ordinary_gradient_descent(self, inputs, pred, alpha, _lambda = 0, dropout = []):
        cost = self.forward_propagation(inputs, pred, dropout)
        self.backward_propagation()
        for i in range(len(self.layers)):
            self.layers[i].theta -= alpha * (self.layers[i].derivative +  _lambda * self.regularization(self.layers[i].theta))
        return cost

    @numba.jit(forceobj =  True)
    def stochastic_gradient_descent(self, inputs, pred, alpha, _lambda = 0, dropout = []):
        cost = 0
        for i in range(inputs.shape[0]):
            cost += self.forward_propagation(inputs[i,:].reshape(1, inputs.shape[1]), pred[i,:].reshape(1, pred.shape[1]), error, dropout) / (inputs.shape[0])
            self.backward_propagation()
            for i in range(len(self.layers)):
                self.layers[i].theta -= alpha * (self.layers[i].derivative +  _lambda * self.regularization(self.layers[i].theta))
        return cost

    @numba.njit(forceobj = True)
    def dqn_gradient_descent(self, state, pred, error, dropout = [], beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0):
        #If not done already, initialize the appropriate lists of matrices
        if self.first_momentum == None:
            self.first_momentum, self.second_momentum = [], []
            self.t = 1
            for i in range(len(self.layers)):
                self.first_momentum.append(np.zeros(self.layers[i].theta.shape))
                self.second_momentum.append(np.zeros(self.layers[i].theta.shape)) 
        #This function takes one gradient descent step using whole momentum (aka Adam), while only considering updating the desired action
        self.forward_propagation(state, pred, error, dropout)
        #In a dqn, we only consider the output for our desired action, which is the only non-zero entry in pred
        self.layers[-1].output *= pred != 0
        cost = self.layers[-1].error(self.layers[-1].output, self.pred)
        self.backward_propagation()
        for i in range(len(self.layers)):
            self.first_momentum[i] = beta_1 * self.first_momentum[i] + (1 - beta_1) * self.layers[i].theta
            self.second_momentum[i] = beta_2 * self.second_momentum[i] + (1 - beta_2) * np.power(self.layers[i].theta, 2)
            m_hat = self.first_momentum[i] / (1 - np.power(beta_1, self.t))
            v_hat = self.second_momentum[i] / (1 - np.power(beta_2, self.t))
            self.layers[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.layers[i].theta.shape) * epsilon) + _lambda * self.regularization(self.layers[i].theta))
        self.t += 1
        return cost

    @numba.njit(forceobj = True)
    def mini_batch_gradient_descent(self, inputs, pred, alpha, _lambda, dropout = [], batchsize = 32):
        p = np.random.permutation(inputs.shape[0])
        batch_X = inputs[p]
        batch_Y = pred[p]
        cost = 0
        for i in range(pred.shape[0] // batchsize):
            cost += self.ordinary_gradient_descent(self, batch_X[i * batchsize:(i+1) * batchsize, :], batch_Y[i * batchsize:(i+1) * batchsize], alpha, _lambda, dropout) / batchsize
        self.ordinary_gradient_descent(self, batch_X[(i+1) * batchsize:, :], batch_Y[(i+1) * batchsize:], alpha, _lambda, dropout) / (inputs.shape[0] * (pred.shape[0] // batchsize))
        return cost

    def adam(self, inputs, pred, error, dropout = [], batchsize = 32, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0):
        p = np.random.permutation(inputs.shape[0])
        batch_X = inputs[p]
        batch_Y = pred[p]
        cost = 0
        #Standard update rule for Adam (Gradient descent w/ momentum)
        for i in range(pred.shape[0] // batchsize):
            cost += self.forward_propagation(batch_X[i * batchsize:(i+1) * batchsize, :], batch_Y[i * batchsize:(i+1) * batchsize], error, dropout) / (inputs.shape[0] * (pred.shape[0] // batchsize))
            self.backward_propagation()
            for i in range(len(self.theta_grad)):
                self.first_momentum[i] = beta_1 * self.first_momentum[i] + (1 - beta_1) * self.theta_grad[i]
                self.second_momentum[i] = beta_2 * self.second_momentum[i] + (1 - beta_2) * np.power(self.theta_grad[i], 2)
                m_hat = self.first_momentum[i] / (1 - np.power(beta_1, self.t))
                v_hat = self.second_momentum[i] / (1 - np.power(beta_2, self.t))
                #The last part in the denominator is purely to avoid division by zero 
                self.weights[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[i].theta.shape[0], self.weights[i].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
            self.t += 1
        
        #Repeating this for the last batch 
        cost += self.forward_propagation(batch_X[(i+1) * batchsize:, :], batch_Y[(i+1) * batchsize:], error, dropout) / (inputs.shape[0] * (pred.shape[0] // batchsize))
        self.backward_propagation()
        for i in range(len(self.theta_grad)):
            self.first_momentum[i] = beta_1 * self.first_momentum[i] + (1 - beta_1) * self.theta_grad[i]
            self.second_momentum[i] = beta_2 * self.second_momentum[i] + (1 - beta_2) * np.power(self.theta_grad[i], 2)
            m_hat = self.first_momentum[i] / (1 - np.power(beta_1, self.t))
            v_hat = self.second_momentum[i] / (1 - np.power(beta_2, self.t))
            self.weights[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[i].theta.shape[0], self.weights[i].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        self.t += 1
        return cost

    #Auxilliary methods:
    def regularization(self, theta):
        return np.vstack((np.zeros((1, theta.shape[1])), theta[1:, :]))
    
    #NOT WORKING AS INTENDED YET (need to enable deep copies)
    def retrieve_weights(self):
        return self.layers.copy()
    #NOT WORKING AS INTENDED YET (need to enable deep copies)
    def assign_weights(self, new_layers):
        self.layers = new_layers.copy()

    def load_weights(self, path):
        with open(str(path), "rb") as layers:
            layers = pickle.load(layers)
        self.assign_weights(layers)

    def save_weights(self, path):  
        with open(str(path), 'wb') as layers:
            pickle.dump(self.retrieve_weights(), layers, pickle.HIGHEST_PROTOCOL)
    
    def onehotencode_along_axis(self, arr, axis):
        # Setup o/p hot encoded int array 
        h = np.zeros(arr.shape)
        idx = arr.argmax(axis=axis)

        # Setup same dimensional indexing array as the input
        idx = np.expand_dims(idx, axis)
        # Finally assign True values
        np.put_along_axis(h,idx,1,axis=axis)
        return h
    
    def round_array(self, arr):
        return np.around(arr)

    def accuracy(self, y_hat, y):
        if len(y_hat.shape) == 1:
            y_hat.reshape(y_hat.shape[0], 1)
        if len(y.shape) == 1:
            y.reshape(y.shape[0], 1)
        y_hat = self.round_array(y_hat)
        return np.sum(np.all(np.equal(y_hat, y), axis=1))/y_hat.shape[0]

    def output_layer(self):
        return self.layers[-1].output