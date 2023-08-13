#implementing a standard NN in a "nice", object oriented way to kinda learn that or smth
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numba
from skimage.util.shape import view_as_windows

class neural_network(ABC):
    @abstractmethod
    def forward_propagation(self):
        raise NotImplementedError
    @abstractmethod
    def backward_propagation(self):
        raise NotImplementedError

class transformer(neural_network):
    pass


#Current project
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
    
    def retrieve_weights(self):
        return self.layers.copy()

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

#"Naive" finished implementation of an ordinary neural network, past project
class cont_feedforward_nn(neural_network):
    class weight:
        def __init__(self, input_size, output_size):
            self.theta = np.random.rand(input_size, output_size)
            e = np.sqrt(2) / np.sqrt(self.theta.shape[0])
            self.theta = self.theta * 2 * e - e
    class layer:
        @numba.jit(forceobj = True)
        def __init__(self, weights, prevLayer, non_linear):
            self.z = prevLayer @ weights
            self.a = np.hstack((np.ones((self.z.shape[0], 1)), non_linear(self.z)))

    def __init__(self, input_size, nodes, non_linear, non_linear_derivative, output, output_deriv, len_output):
        self.input_size = input_size
        self.nodes = nodes
        self.non_linear = non_linear
        self.non_linear_derivative = non_linear_derivative
        self.output = output
        self.len_output = len_output
        self.output_deriv = output_deriv

        self.weights = []
        prev = input_size
        for i in range(len(self.nodes)):
            self.weights.append(self.weight(prev + 1, nodes[i]))
            prev = nodes[i]
        self.weights.append(self.weight(prev + 1, len_output))

        #Specifically for Adam:
        self.first_momentum = [np.zeros((i.theta.shape[0], i.theta.shape[1])) for i in self.weights]
        self.second_momentum = [np.zeros((i.theta.shape[0], i.theta.shape[1])) for i in self.weights]
        self.t = 1

    class outputs:
        @numba.jit(forceobj = True)
        def __init__ (self, weights, prevLayer, output):
            self.z = output(prevLayer @ weights)
    @numba.jit(forceobj = True)
    def forward_propagation(self, inputs, pred, error, dropout = []):
        dropout = dropout.copy()
        self.pred = np.array(pred)
        if inputs.shape[0] == 1:
            inputs.reshape(1, inputs.shape[1])
        if len(inputs.shape) == 1:
            inputs.reshape(inputs.shape[0], 1)
        self.layers_for = []
        self.layer_input = [np.hstack((np.ones((inputs.shape[0], 1)), inputs))]
        for i in range(len(self.nodes)):
            self.layers_for.append(self.layer(self.weights[i].theta, self.layer_input[-1], self.non_linear))
            if len(dropout) > 0:
                dropout_percentage = dropout.pop(0)
                layer_dropout = np.random.rand(self.layers_for[-1].z.shape[0], self.layers_for[-1].z.shape[1])
                layer_dropout = (layer_dropout > (1-dropout_percentage)) / (1-dropout_percentage)
                self.layers_for[-1].a *= np.hstack((np.ones((layer_dropout.shape[0], 1)), layer_dropout)) 
            self.layer_input.append(self.layers_for[-1].a)
        self.layers_for.append(self.outputs(self.weights[-1].theta, self.layer_input[-1], self.output))
        return error(self.layers_for[-1].z, self.pred)
    @numba.jit(forceobj = True)
    def backward_propagation(self):
        self.delta = [self.output_deriv(self.layers_for[-1].z, self.pred.reshape(self.layers_for[-1].z.shape))]
        for i in range(1, len(self.layers_for)):
            self.delta.append(self.delta[-1] @ self.weights[-i].theta[1:, :].T * self.non_linear_derivative(self.layers_for[-i - 1].z))
        self.theta_grad = []
        for i in range(len(self.delta)):
            self.theta_grad.append(self.layer_input[i].T @ self.delta[::-1][i]) 
        return self.theta_grad
    @numba.jit(forceobj = True)
    def ordinary_gradient_descent(self, alpha, _lambda, inputs, pred, error, dropout = []):
        cost = self.forward_propagation(inputs, pred, error, dropout)
        self.backward_propagation()
        for i in range(len(self.weights)):
            self.weights[i].theta -= alpha/inputs.shape[0] * (self.theta_grad[i] + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        return cost
    @numba.jit(forceobj = True)
    def stochastic_gradient_descent(self, alpha, _lambda, inputs, pred, error, dropout = []):
        cost = 0
        for i in range(inputs.shape[0]):
            cost += self.forward_propagation(inputs[i,:].reshape(1, inputs.shape[1]), pred[i,:].reshape(1, pred.shape[1]), error, dropout) / (2*inputs.shape[0])
            self.backward_propagation()
            for i in range(len(self.weights)):
                self.weights[i].theta -= alpha * (self.theta_grad[i] + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        return cost
    @numba.jit(forceobj = True)
    def dqn_gradient_descent(self, state, pred, error, dropout = [], batchsize = 32, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0):
        #This function takes one gradient descent step using momentum (aka Adam), while only considering updating the desired action
        self.forward_propagation(state, pred, error, dropout)
        #In a dqn, we only consider the output for our desired action, which is the only non-zero entry in pred
        self.layers_for[-1].z *= pred != 0
        cost = error(self.layers_for[-1].z, self.pred)
        self.backward_propagation()
        for i in range(len(self.theta_grad)):
            self.first_momentum[i] = beta_1 * self.first_momentum[i] + (1 - beta_1) * self.theta_grad[i]
            self.second_momentum[i] = beta_2 * self.second_momentum[i] + (1 - beta_2) * np.power(self.theta_grad[i], 2)
            m_hat = self.first_momentum[i] / (1 - np.power(beta_1, self.t))
            v_hat = self.second_momentum[i] / (1 - np.power(beta_2, self.t))
            self.weights[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[i].theta.shape[0], self.weights[i].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        self.t += 1
        return cost
    @numba.jit(forceobj = True)
    def mini_batch_gradient_descent(self, alpha, _lambda, inputs, pred, error, dropout = [], batchsize = 32):
        p = np.random.permutation(inputs.shape[0])
        batch_X = inputs[p]
        batch_Y = pred[p]
        cost = 0
        for i in range(pred.shape[0] // batchsize):
            cost += self.ordinary_gradient_descent(self, alpha, _lambda, batch_X[i * batchsize:(i+1) * batchsize, :], batch_Y[i * batchsize:(i+1) * batchsize], error, dropout) / batchsize
        self.ordinary_gradient_descent(self, alpha, _lambda, batch_X[(i+1) * batchsize:, :], batch_Y[(i+1) * batchsize:], error, dropout) / (inputs.shape[0] * (pred.shape[0] // batchsize))
        return cost
    @numba.jit(forceobj = True)
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
    @numba.jit(forceobj = True)
    def reset_adam(self):
        self.first_momentum = [np.zeros((i.theta.shape[0], i.theta.shape[1])) for i in self.weights]
        self.second_momentum = [np.zeros((i.theta.shape[0], i.theta.shape[1])) for i in self.weights]
        self.t = 1
    @numba.jit(forceobj = True)
    def adam_iterated(self, inputs, pred, error, dropout = [], batchsize = 32, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0, iterations = 150):
        self.reset_adam()
        for i in range(iterations):
            loss = self.adam(inputs, pred, error, dropout, batchsize, beta_1, beta_2, epsilon, alpha, _lambda)
        return loss
    def BFGS(self):
        #direction = -H0 @ grad(xk)
        #perform line search to obtain ak, satisfying wolf conditions
        #xk+1 = xk + ak@pk
        #yk = grad(xk+1) - grad(xk)
        #Hk+1 = Hk + (sk.T@yk + yk.T @ Hk@yk)*(sk@sk.T)/(sk.T@yk)^2 - (Hk@yk@sk.T + sk@yk.T@Hk)/(sk.T @ yk) 
        return 0

    def early_stopping(self, cross_val):
        self.last_error = 1000000000
        self.cross_val = cross_val

    def retrieve_weights(self):
        return self.weights.copy()

    def assign_weights(self, new_weights):
        self.weights = new_weights.copy()

    def load_weights(self, path):
        with open(str(path), "rb") as weights:
            weights = pickle.load(weights)
        self.assign_weights(weights)

    def save_weights(self, path):
        with open(str(path), 'wb') as weights:
            pickle.dump(self.retrieve_weights(), weights, pickle.HIGHEST_PROTOCOL)
    
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
        print(y_hat, y)
        return np.sum(np.all(np.equal(y_hat, y), axis=1))/y_hat.shape[0]

    def output_layer(self):
        return self.layers_for[-1].z    

class autoencoder(cont_feedforward_nn):
    def __init__(self, input_size, nodes, non_linear, non_linear_derivative, encoding_layer):
        self.input_size = input_size
        self.nodes = nodes
        self.non_linear = non_linear
        self.non_linear_derivative = non_linear_derivative
        self.output = sigmoid
        self.len_output = input_size
        self.output_deriv = Sigmoid_out_deriv
        self.encoding_layer = encoding_layer

        self.weights = []
        prev = input_size
        for i in range(len(self.nodes)):
            self.weights.append(self.weight(prev + 1, nodes[i]))
            prev = nodes[i]
        self.weights.append(self.weight(prev + 1, self.len_output))

    def forward_propagation(self, inputs):
        self.pred = np.array(inputs)
        self.layers_for = []
        self.layer_input = [np.hstack((np.ones((inputs.shape[0], 1)), inputs))]
        for i in range(len(self.nodes)):
            self.layers_for.append(self.layer(self.weights[i].theta, self.layer_input[-1], self.non_linear))
            self.layer_input.append(self.layers_for[-1].a)
        self.layers_for.append(self.outputs(self.weights[-1].theta, self.layer_input[-1], self.output))
        return MSE(self.layers_for[-1].z, self.pred)

    def backward_propagation(self):
        self.delta = [self.output_deriv(self.layers_for[-1].z, self.pred.reshape(self.layers_for[-1].z.shape))]
        for i in range(1, len(self.layers_for)):
            self.delta.append(self.delta[-1] @ self.weights[-i].theta[1:, :].T * self.non_linear_derivative(self.layers_for[-i - 1].z))
        self.theta_grad = []
        for i in range(len(self.delta)):
            self.theta_grad.append(self.layer_input[i].T @ self.delta[::-1][i]) 
        return self.theta_grad

    def ordinary_gradient_descent(self, alpha, inputs, _lambda):
        cost = self.forward_propagation(inputs)
        self.backward_propagation()
        for i in range(len(self.weights)):
            self.weights[i].theta -= alpha/inputs.shape[0] * (self.theta_grad[i] + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        return cost

    def stochastic_gradient_descent(self, alpha, inputs, _lambda):
        cost = 0
        for i in range(inputs.shape[0]):
            cost += self.forward_propagation(inputs[i,:].reshape(1, inputs.shape[1])) / inputs.shape[0]
            self.backward_propagation()
            for i in range(len(self.weights)):
                self.weights[i].theta -= alpha * (self.theta_grad[i] + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :])))
        return cost

    def encode(self, inputs):
        self.layers_for = []
        self.layer_input = [np.hstack((np.ones((inputs.shape[0], 1)), inputs))]
        for i in range(self.encoding_layer + 1):
            self.layers_for.append(self.layer(self.weights[i].theta, self.layer_input[-1], self.non_linear))
            self.layer_input.append(self.layers_for[-1].a)
        return self.layers_for[-1].z
    def decode(self, inputs):
        self.layers_for = []
        self.layer_input = [np.hstack((np.ones((inputs.shape[0], 1)), inputs))]
        for i in range(1, self.layers - self.encoding_layer + 1):
            self.layers_for.append(self.layer(self.weights[self.encoding_layer + i].theta, self.layer_input[-1], self.non_linear))
            self.layer_input.append(self.layers_for[-1].a)
        return self.output(self.layers_for[-1].z)

    def visualize_results(self, X, rows, columns):
        f, axarr = plt.subplots(rows, columns * 2)
        for i in range(rows):
            for j in range(columns):
                output = self.encode(X[i + rows*j].reshape(1, 784))
                image = self.decode(output)
                axarr[i, 2*(j)].imshow(X[i + rows*j].reshape(28, 28), cmap = "gray", vmin=0,vmax=1)
                axarr[i, 2*(j) + 1].imshow(image[0, :].reshape(28, 28), cmap = "gray", vmin=0,vmax=1)
        plt.show()

class RNN_many_to_many(cont_feedforward_nn):
    def __init__(self, input_size, nodes, is_recurrent, non_linear, non_linear_derivative, output, output_deriv, len_output):
        self.input_size = input_size
        self.nodes = nodes
        self.non_linear = non_linear
        self.non_linear_derivative = non_linear_derivative
        self.output = output
        self.len_output = len_output
        self.output_deriv = output_deriv
        self.is_recurrent = is_recurrent

        if len(is_recurrent) != len(self.nodes):
            raise ValueError("Need to specify if a layer is recurrent for every layer.")

        self.weights = []
        prev = input_size
        for i in range(len(self.nodes)):
            self.weights.append(self.weight(prev + 1, nodes[i]))
            if is_recurrent[i] == 1:
                self.weights[-1].theta = np.vstack((self.weight(prev, prev).theta,self.weights[i].theta))
            else:
                self.weights[-1].theta = np.vstack((np.zeros((prev, prev)), self.weights[i].theta))
            prev = nodes[i]
        self.weights.append(self.weight(prev + 1, len_output))
    
    def forward_propagation(self, inputs, pred, error, dropout = []):
        #Every row represents one datapoint in the time series, each column the corresponding features
        #hstacked to the columns is the "memory" of the rnn
        #In essence, we do stochastic gradient descent on every "string" of time-series data
        self.pred = np.array(pred)
        self.layers_for = []
        self.layer_input = [np.hstack((np.zeros((inputs[0].shape[1], 0)), np.ones((inputs[0].shape[1], 0)), inputs[0]))]
        
        for i in range(len(self.nodes)):
            for j in range(self.layer_inputs[-1].shape[0]):
                self.layers_for.append(self.layer(self.weights[i].theta, self.layer_input[-1], self.non_linear))
                if len(dropout) > 0:
                    layer_dropout = np.random.rand(self.layers_for[-1].z.shape[0], self.layers_for[-1].z.shape[1])
                    dropout_percentage = dropout.pop(0)
                    self.layers_for[-1].a *= np.hstack((np.ones((layer_dropout.shape[0], 1)), layer_dropout > (1-dropout_percentage) / (1-dropout_percentage))) 
            self.layer_input.append(self.layers_for[-1].a)
        self.layers_for.append(self.outputs(self.weights[-1].theta, self.layer_input[-1], self.output))
        return error(self.layers_for[-1].z, self.pred)

class RNN_one_to_many(RNN_many_to_many):
    def __init__(self):
        pass

def k_fold_cross_val(neural_net, net_error, train_X, train_Y, k = 10):
    subsample_size = train_X.shape[0] // k
    p = np.random.permutation(train_X.shape[0])
    shuffle_X = train_X[p]
    shuffle_Y = train_Y[p]
    cross_val_error = 0
    for i in range(k - 1):
        cross_val_error += neural_net.forward_propagation(train_X[i*subsample_size:(i+1)*subsample_size, :], train_Y[i*subsample_size:(i+1)*subsample_size], net_error) / k
    cross_val_error += neural_net.forward_propagation(train_X[(i+1)*subsample_size:, :], train_Y[(i+1)*subsample_size:], net_error) / k
    return cross_val_error 

def sigmoid(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def sigmoidDeriv(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig * (1-sig)

def ReLU(Z):
    return np.maximum(Z,0)

def ReLUDeriv(Z):
    return Z > 0

def output(Z):
    return Z

def MSE(pred, Y):
    return np.sum((np.array(pred) - np.array(Y).reshape(np.array(pred).shape))**2) / (2 * np.array(pred).size)

def ReLU_out_deriv(x, y):
    return x - y * (x - y > 0)

def MSE_out_deriv(x, y):
    return (x - y) / x.shape[0]

def Sigmoid_out_deriv(x, y):
    out = x - y 
    return out #* sigmoidDeriv(out)

def logistic_cost(y_hat, y):
    return np.sum(-y * np.log2(y_hat + (y_hat == 0)*0.0001) - (1-y) * np.log2(1 - y_hat + (y_hat == 1)*0.0001)) / y.shape[0]

def softmax(x):
    x -= x.max()
    sum = np.sum(np.exp(x), axis = 1)
    for i in range(x.shape[1]):
        x[:, i] = np.exp(x[:, i]) / sum
    return x

def softmaxDeriv(x, y):
    pass

def const(Z):
    return np.ones(Z.shape)

def get_info():
    print("In order to build a neural network, you have to call the cont_feedforward_nn class.")
    print("There, you have to specify:")
    print("The input size (i.e. how many inputs do you have)")
    print("A list with the amount of neurons for each hidden layer")
    print("The non-linear function used in the hidden layers: ReLU or Sigmoid")
    print("The derivative of the non-linear function: ReLUDeriv or SigmoidDeriv")
    print("The output function: output, sigmoid or softmax")
    print("The derivative of the output function: MSESigmoid or softmax")
    print("The amount of outputs: (length of the output vector)")