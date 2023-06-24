from keras.datasets import mnist
import NN
import matplotlib.pyplot as plt
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]) / 255
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2])) / 255
encoding_size = 2
autoencoder = NN.autoencoder(784, 5, [300, 100, encoding_size, 100, 300], NN.ReLU, NN.ReLUDeriv, 2)
autoencoder.load_weights("D:\Damian\PC\Python\ML\Autoencoder\weights.pkl")
for i in range(0):
    print(autoencoder.stochastic_gradient_descent(0.00001, train_X, 0.0))
    #autoencoder.save_weights("D:\Damian\PC\Python\ML\Autoencoder\weights.pkl")
autoencoder.visualize_results(test_X, 20, 10)


