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

#"Naive" finished implementation of an ordinary neural network, past project
class cont_feedforward_nn(neural_network):
    class weight:
        def __init__(self, input_size, output_size, non_linear = None):
            if non_linear == ReLU:
                self.theta = np.random.normal(0 ,np.sqrt(2) / np.sqrt(input_size) , size = (input_size, output_size))
            if non_linear == sigmoid:
                self.theta = np.random.uniform(-np.sqrt(6) / np.sqrt(input_size + output_size), np.sqrt(6) / np.sqrt(input_size + output_size), size = (input_size, output_size))
            else:
                self.theta = np.random.rand(input_size, output_size) * np.sqrt(2) / np.sqrt(input_size) * 2 - np.sqrt(2) / np.sqrt(input_size)
        def copy(self):
            l = __class__(self.theta.shape[0], self.theta.shape[1])
            l.theta = self.theta.copy()
            return l
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
            self.weights.append(self.weight(prev + 1, nodes[i], self.non_linear))
            prev = nodes[i]
        self.weights.append(self.weight(prev + 1, len_output, self.output))

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
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(inputs.shape[0], 1)
        self.layers_for = []
        self.layer_input = [np.hstack((np.ones((inputs.shape[0], 1)), inputs))]
        for i in range(len(self.nodes)):
            self.layers_for.append(self.layer(self.weights[i].theta, self.layer_input[-1], self.non_linear))

            #Drop-out implementation: ----------------------------------------------------------------------
            if len(dropout) > 0:
                dropout_percentage = dropout.pop(0)
                layer_dropout = np.random.rand(self.layers_for[-1].z.shape[0], self.layers_for[-1].z.shape[1])
                layer_dropout = (layer_dropout > (1-dropout_percentage)) / (1-dropout_percentage)
                self.layers_for[-1].a *= np.hstack((np.ones((layer_dropout.shape[0], 1)), layer_dropout)) 
            #------------------------------------------------------------------------------------------------
                
            self.layer_input.append(self.layers_for[-1].a)
        self.layers_for.append(self.outputs(self.weights[-1].theta, self.layer_input[-1], self.output))
        return error(self.layers_for[-1].z, self.pred)
    @numba.jit(forceobj = True)
    def backward_propagation(self):
        #Derivative w.r.t. output of last linear layer
        self.delta = [self.output_deriv(self.layers_for[-1].z, self.pred.reshape(self.layers_for[-1].z.shape))]
        #Chain-rule implementation
        for i in range(1, len(self.layers_for)):
            #Derivative w.r.t. 
            self.delta.append(self.delta[-1] @ self.weights[-i].theta[1:, :].T * self.non_linear_derivative(self.layers_for[-i - 1].z))
        self.theta_grad = []
        for i in range(len(self.delta)):
            self.theta_grad.append(self.layer_input[i].T @ self.delta[-i - 1])
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
            self.weights[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[i].theta.shape[0], self.weights[i].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :]))) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
        self.t += 1
        return cost
    @numba.jit(forceobj = True)
    def mini_batch_gradient_descent(self, alpha, _lambda, inputs, pred, error, dropout = [], batchsize = 32):
        p = np.random.permutation(inputs.shape[0])
        batch_X = inputs[p]
        batch_Y = pred[p]
        cost = 0
        for i in range(pred.shape[0] // batchsize):
            cost += self.ordinary_gradient_descent(self, alpha, _lambda, batch_X[i * batchsize:(i+1) * batchsize, :], batch_Y[i * batchsize:(i+1) * batchsize], error, dropout) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
        cost += self.ordinary_gradient_descent(self, alpha, _lambda, batch_X[(i+1) * batchsize:, :], batch_Y[(i+1) * batchsize:], error, dropout) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
        return cost
    @numba.jit(forceobj = True)
    def adam(self, inputs, pred, error, dropout = [], batchsize = 32, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0.0):
        p = np.random.permutation(inputs.shape[0])
        batch_X = inputs[p]
        batch_Y = pred[p]
        cost = 0
        #Standard update rule for Adam (Gradient descent w/ momentum)
        for i in range(pred.shape[0] // batchsize):
            cost += self.forward_propagation(batch_X[i * batchsize:(i+1) * batchsize, :], batch_Y[i * batchsize:(i+1) * batchsize], error, dropout) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
            self.backward_propagation()
            for j in range(len(self.theta_grad)):
                self.first_momentum[j] = beta_1 * self.first_momentum[j] + (1 - beta_1) * self.theta_grad[j]
                self.second_momentum[j] = beta_2 * self.second_momentum[j] + (1 - beta_2) * np.power(self.theta_grad[j], 2)
                m_hat = self.first_momentum[j] / (1 - np.power(beta_1, self.t))
                v_hat = self.second_momentum[j] / (1 - np.power(beta_2, self.t))
                #The last part in the denominator is purely to avoid division by zero 
                self.weights[j].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[j].theta.shape[0], self.weights[j].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[j].theta.shape[1])), _lambda*self.weights[j].theta[1:, :]))) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
            self.t += 1
        
        if pred.shape[0] < batchsize:
            i = -1
        #Repeating this for the last batch 
        cost += self.forward_propagation(batch_X[(i+1) * batchsize:, :], batch_Y[(i+1) * batchsize:], error, dropout) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
        self.backward_propagation()
        for i in range(len(self.theta_grad)):
            self.first_momentum[i] = beta_1 * self.first_momentum[i] + (1 - beta_1) * self.theta_grad[i]
            self.second_momentum[i] = beta_2 * self.second_momentum[i] + (1 - beta_2) * np.power(self.theta_grad[i], 2)
            m_hat = self.first_momentum[i] / (1 - np.power(beta_1, self.t))
            v_hat = self.second_momentum[i] / (1 - np.power(beta_2, self.t))
            self.weights[i].theta -= alpha * ((m_hat) / (np.sqrt(v_hat) + np.random.rand(self.weights[i].theta.shape[0], self.weights[i].theta.shape[1]) * epsilon) + np.vstack((np.zeros((1, self.weights[i].theta.shape[1])), _lambda*self.weights[i].theta[1:, :]))) / (pred.shape[0] // batchsize + int(pred.shape[0] % batchsize != 0))
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

    def retrieve_weights(self):
        weight_list = []
        for i in self.weights:
            weight_list.append(i.copy())
        return weight_list

    def assign_weights(self, new_weights):
        self.weights = []
        for i in new_weights:
            self.weights.append(i)

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
        return self.layers_for[-1].z.copy()
    
    def predict(self, features):
        self.forward_propagation(self, features, np.zeros((features.shape[0], self.len_output)), lambda a, b : 0, dropout = [])    
        return self.output_layer()
    
    @numba.jit(forceobj = True)
    def early_stopping_adam_iterated(self, inputs, pred, error, dropout = [], batchsize = 32, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, alpha = 0.001, _lambda = 0, iterations = 150, val_perc = 0.1, seed = 0):
        self.reset_adam()
        np.random.seed(0)
        p = np.random.permutation(inputs.shape[0])
        features, labels = inputs[p], pred[p]
        val_features, val_labels = features[:int(inputs.shape[0] * val_perc), :], labels[:int(inputs.shape[0] * val_perc), :]
        train_features, train_labels = features[int(inputs.shape[0] * val_perc):, :], labels[int(inputs.shape[0] * val_perc):, :]
        #Uses early stopping when the generalization error increases for s consecutive strips
        best_weights, best_score = self.retrieve_weights(), 100000000
        memory = [] #Queue for our cross validation score: 
        for i in range(iterations):
            loss = self.adam(train_features, train_labels, error, dropout, batchsize, beta_1, beta_2, epsilon, alpha, _lambda)
            #Saving the best weights
            val_score = self.forward_propagation(val_features, val_labels, error)
            if val_score < best_score:
                best_weights, best_score = self.retrieve_weights(), val_score
            #We are using UP3 from https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf, and start this process after a "warm-up" of 100 iterations
            #A 'training strip' of length k is a sequence of k epochs n + 1, ..., n + k where n is divisible by k, and we measure after each strip
            #We set k = 3 epochs, and stop after three strips if, between each strip, the value always increased: UP_3 == True if (UP_2 and E_va(t) > E_va(t - k)), UP_2 == True if (UP_1 and E_va(t - k) > E_va(t - 2k)), UP_1 == True if E_va(t - 2k) > E_va(t - 3k)
            
            #Adding the strip values to the queue   
            if i % 3 == 0:
                memory.insert(0, val_score)
                if len(memory) > 4:
                    memory.pop()
                    #Checking if UP_3 == True (one could have done this way better, but idc)
                    if i > 50:
                        up_1 = (memory[3] < memory[2])
                        up_2 = (memory[2] < memory[1]) and up_1
                        up_3 = (memory[1] < memory[0]) and up_2
                        if up_3:
                            break
        self.assign_weights(best_weights)
        return best_score

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

class ActivationFunction():
    def __init__(self, func, deriv) -> None:
        self.func = func
        self.deriv = deriv
    
class ErrorFunction():
    def __init__(self) -> None:
        pass

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
    return (x - y)

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

def mape(y_hat, y):
    return np.sum( np.abs((np.array(y_hat) - np.array(y).reshape(np.array(y_hat).shape)) / np.array(y_hat)) ) / (np.array(y_hat).size)

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