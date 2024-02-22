import NN
from keras.datasets import mnist
import numpy as np
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]) / 255
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2])) / 255
#One-Hot encoding
train_Y = np.zeros((train_y.size, train_y.max() + 1))
train_Y[np.arange(train_y.size), train_y] = 1
test_Y = np.zeros((test_y.size, test_y.max() + 1))
test_Y[np.arange(test_y.size), test_y] = 1

neural_net_adam = NN.cont_feedforward_nn(784, [200, 200], NN.ReLU, NN.ReLUDeriv, NN.sigmoid, NN.Sigmoid_out_deriv, train_y.max() + 1)
neural_net_sgd = NN.cont_feedforward_nn(784, [200, 200], NN.ReLU, NN.ReLUDeriv, NN.sigmoid, NN.Sigmoid_out_deriv, train_y.max() + 1)

for i in range(5):
    print(neural_net_adam.adam(train_X, train_Y, NN.logistic_cost))
    #print(neural_net_sgd.stochastic_gradient_descent(0.001, 0, train_X, train_Y, NN.logistic_cost))
neural_net_adam.forward_propagation(train_X, train_Y, NN.logistic_cost)
print(neural_net_adam.accuracy(neural_net_adam.output_layer(), train_Y))