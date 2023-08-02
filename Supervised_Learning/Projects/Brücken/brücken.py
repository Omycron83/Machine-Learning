import NN
import xgboost
import pandas as pd
import numpy as np
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken_NN\DatensatzTraining.csv")
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
train_data = data.to_numpy()
train_data = np.delete(train_data, 1, 1)
neuronales_netz = NN.cont_feedforward_nn(6, [60], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)

def k_fold_cross_val_nn(neuralnet, k, data, labels, alpha, _lambda, error_func):
    weights = neuralnet.retrieve_weights().copy()
    p = np.random.permutation(data.shape[0])
    shuffle_data = data[p]
    shuffle_labels = labels[p]
    error = 0
    for l in range(k - 1):
        neuronales_netz.assign_weights(weights)
        print(neuralnet.forward_propagation(shuffle_data[data.shape[0] // k * l:data.shape[0] // k * (l+1), :], shuffle_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :], error_func) / k)
        for i in range(20):
            neuronales_netz.stochastic_gradient_descent(alpha, _lambda, np.vstack((shuffle_data[0 : data.shape[0] // k * l, :], shuffle_data[data.shape[0] // k * (l + 1):, :])), np.vstack((shuffle_labels[0 : data.shape[0] // k * l, :], shuffle_labels[data.shape[0] // k * (l + 1):, :])), error_func)
        error += neuralnet.forward_propagation(shuffle_data[data.shape[0] // k * l:data.shape[0] // k * (l+1), :], shuffle_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :], error_func) / k
    
    neuronales_netz.assign_weights(weights)
    for i in range(20):
        neuronales_netz.stochastic_gradient_descent(alpha, _lambda, shuffle_data[0 : data.shape[0] // k * (l + 1), :], shuffle_labels[0 : data.shape[0] // k * (l + 1), :], error_func)
    error += neuralnet.forward_propagation(shuffle_data[data.shape[0] // k * (l + 1):, :], shuffle_labels[data.shape[0] // k * (l + 1):, :], error_func) / k
    
    return error



neuronales_netz.forward_propagation(train_data, labels, NN.MSE)
for i in range(10):
    print(i + 1, ":", neuronales_netz.stochastic_gradient_descent(0.001, 0, train_data, labels, NN.MSE))

print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.1, 0, NN.MSE))
print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.01, 0, NN.MSE))
print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.001, 0, NN.MSE))
print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.0001, 0, NN.MSE))
print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.00001, 0, NN.MSE))
print(k_fold_cross_val_nn(neuronales_netz, 10, train_data, labels, 0.000001, 0, NN.MSE))

