import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
train_X = (train_X - np.average(train_X)) / np.std(train_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
test_X = (test_X - np.average(train_X)) / np.std(train_X)

model = MLPClassifier(solver="lbfgs", alpha = 0.0001, hidden_layer_sizes=(100, 100), activation="relu")
model.fit(train_X, train_y)
model.predict(train_X)