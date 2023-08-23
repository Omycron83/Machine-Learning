#Trying to figure out a decent early-stopping setup with an approximate, though suboptimale neural network
nn_opt = [555, 0.5119633527495826, 47, 0.005015370459087969, 5.0] #0.0005015370459087969
import numpy as np
import pandas as pd
import NN
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Va.csv")
data["BETA_HT_Q_DEB"] = data["BETA_HT_Q_DEB"].fillna(data["BETA_HT_Q_DEB"].mean())
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
train_data = data.to_numpy()
train_data = np.delete(train_data, 1, 1)

nn_one = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
nn_two = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
weights = nn_one.retrieve_weights()

np.random.seed(0)
g = np.random.permutation(train_data.shape[0])
x, y = train_data[g], labels[g]
val_features, val_labels = x[:int(x.shape[0] * 0.1), :], y[:int(y.shape[0] * 0.1), :]
train_features, train_labels = x[int(x.shape[0] * 0.1):, :], y[int(y.shape[0] * 0.1):, :]


val = []
train = []
for i in range(300):
    value = nn_one.adam(train_features, train_labels, NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4])
    train.append(value)
    val.append(nn_one.forward_propagation(val_features, val_labels, NN.MSE))

#print(val)
#print("----------------------------")
#print(train)
plt.plot(np.arange(len(val)), train, label = "Train")
plt.plot(np.arange(len(val)), val, label = "Val")
plt.ylim(0, 40)
plt.xlim(0, 300)
plt.legend()
plt.show()
print(min(val[50:200]))
print(min(val[200:500]))
nn_two.assign_weights(weights)
print("Normal-value:", min(val))
print("Early stopping value:", nn_two.early_stopping_adam_iterated(train_data, labels, NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations=300))