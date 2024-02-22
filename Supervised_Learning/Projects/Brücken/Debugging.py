import NN
import pandas as pd
import numpy as np
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Ve.csv")
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
data["BETA_HT_Q_DEB"] = data["BETA_HT_Q_DEB"].fillna(data["BETA_HT_Q_DEB"].mean())
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()
train_data = (train_data - np.mean(train_data, axis = 0)) / np.std(train_data, axis = 0)
"""
gg = NN.cont_feedforward_nn(train_data.shape[1], [5], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
gg.forward_propagation(train_data, labels, NN.MSE)
gg.backward_propagation()
#print(gg.delta[-1])
#fx = gg.layers_for[-2].z
#gg.weights[-2].theta += np.ones(gg.weights[-2].theta.shape) * 0.0000001
#gg.forward_propagation(train_data, labels, NN.MSE)
#fxh = gg.layers_for[-2].z
#print((fx - fxh) / 0.0000001)


print(gg.weights[0].theta, gg.weights[1].theta)

for i in range(10):
    #print(gg.adam(train_data, labels, NN.MSE))
    print(gg.stochastic_gradient_descent(0.001, 0, train_data, labels, NN.MSE))
    #print(gg.theta_grad[0], gg.theta_grad[1])
    #print(gg.weights[0].theta, gg.weights[1].theta)
    #print("-----------------")
#gg.adam_iterated(np.random.rand(1000, 5), np.random.rand(1000, 1), NN.MSE)
gg.forward_propagation(train_data, labels, NN.MSE)

print(gg.output_layer()[:30], gg.layers_for[-2].a[:30], gg.layers_for[-2].z[:30])
print(gg.weights[0].theta, gg.weights[1].theta)

gg = NN.cont_feedforward_nn(1, [400, 400], NN.ReLU, NN.ReLUDeriv, NN.sigmoid, NN.Sigmoid_out_deriv, 1)
for i in range(1):
    gg.adam_iterated(np.array([[5]]), np.array([[0]]), NN.logistic_cost)
gg.forward_propagation(np.array([[5]]), np.array([[0]]), NN.logistic_cost)
print(gg.output_layer())
"""
nn = NN.cont_feedforward_nn(train_data.shape[1], [2560], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
untrained_weights = nn.retrieve_weights()
def train(features, labels):
    nn.assign_weights(untrained_weights)
    nn.reset_adam()
    for i in range(500):
        cost = nn.stochastic_gradient_descent(0.001, 0, features, labels, NN.MSE, dropout=[])
        if i % 5 == 0:
            print(i, cost)
def cost(features, labels):
    nn.forward_propagation(features, labels, NN.MSE)
    print(np.hstack((nn.output_layer()[:30], labels[:30])))
    return NN.mape(labels, nn.output_layer())

train(train_data, labels)
print(cost(train_data, labels))
data_unknown = pd.read_csv(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\data_deployment_Modell_Ve.csv")
data_unknown = data_unknown.to_numpy()
data_id = data_unknown[:, 0]
data_stw = data_unknown[:, 2]
data_unknown = np.delete(data_unknown, 0, 1)
data_unknown = (data_unknown - np.mean(data_unknown, axis = 0)) / np.std(data_unknown, axis = 0)

nn.forward_propagation(data_unknown, np.zeros((data_unknown.shape[0], 1)), NN.MSE)
pred_nn = nn.output_layer()
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\NN.csv", np.hstack((data_id.reshape(-1, 1), pred_nn, data_stw.reshape(-1, 1))), delimiter=",")

import torch
from torch import nn as torch_nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class model(torch_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch_nn.Sequential([torch_nn.Linear(params[0][i], params[0][i + 1]), torch_nn.ReLU(), torch_nn.Dropout(p = params[1]), for i in range(len(params[0]) - 1):])
    def forward(self, x):
        x = torch_nn.Flatten(x) #Just reshapes to shape[0], shape[1] * ... * shape[n]
        return self.network(x)
    
nn = model().to(device)
nn(train_data)
optimizer = optim.Adam(nn.parameters(), lr=params[3], weight_decay=params[4])

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=params[2], shuffle=True)

def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))
torch_nn.MSELoss(reduction='sum')
for id_batch, (x_batch, y_batch) in enumerate(dataloader):
    optimizer.zero_grad()
    y_batch_pred = model(x_batch)
    loss = loss_fn(y_batch_pred, y_batch)
    loss.backward()
    optimizer.step()
