#Final model parameters were:
lin_opt = [3, 0.0315]
nn_opt = [1331, 0, 116, 0.0787, 20]
xgboost_opt = [0.0, 0.19032012543987745, 3, 1100, 1.0, 1, 0.0, 5.0]
 
import xgboost
import NN
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import LinRegr
import pandas as pd
import numpy as np
from torch import nn as torch_nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from copy import deepcopy
#Loading in the data and preparing it
#Loading in the data and preparing it
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_g_Test.csv")
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data_id = data['ID_AUFTRAG_ZPM'].to_numpy().reshape(-1, 1)
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

#Getting the final models score:
def k_fold_cross_val_mapes(k, features, labels, indices, train_func, cost_func, pred, seed = 0):
    #Shuffling
    np.random.seed(seed)
    p = np.random.permutation(features.shape[0])
    shuffled_features = features.copy()[p]
    shuffled_labels = labels.copy()[p]
    indices = indices.copy()[p]
    error = []
    for l in range(k - 1):
        #The test data of the current fold
        test_features = shuffled_features[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        test_labels = shuffled_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        test_indices = indices[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        #The remaining training data of the current fold
        train_features = np.vstack((shuffled_features[:data.shape[0] // k * l, :], shuffled_features[data.shape[0] // k * (l+1):, :]))
        train_labels = np.vstack((shuffled_labels[:data.shape[0] // k * l, :], shuffled_labels[data.shape[0] // k * (l+1):, :])) 
        #Now, train the model on the current fold
        train_func(train_features, train_labels)
        error += [np.hstack((test_indices.reshape(-1, 1), cost_func(test_features, test_labels).reshape(-1, 1), pred(test_features).reshape(-1, 1)))]

    #For the last fold, we dont really know the size of the holdout-set (we dont know about the divisibility of the amount of datapoints by k) so we do this seperately
    #The test data of the last fold
    test_features = shuffled_features[data.shape[0] // k * (l + 1):, :]
    test_labels = shuffled_labels[data.shape[0] // k * (l + 1):, :]
    test_indices = indices[data.shape[0] // k * (l + 1):, :]
    #The remaining training data of the last fold
    train_features = shuffled_features[:data.shape[0] // k * (l + 1), :]
    train_labels = shuffled_labels[:data.shape[0] // k * (l + 1), :]
    #Now, train the model on the current fold
    train_func(train_features, train_labels)
    error += [np.hstack((test_indices.reshape(-1, 1), cost_func(test_features, test_labels).reshape(-1, 1), pred(test_features).reshape(-1, 1)))]
    return np.vstack(error)

lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
def cost(features, labels):
    return np.abs(lin_model.predict(features).reshape(labels.shape[0], 1) - labels) / labels
def pred(features):
    return lin_model.predict(features)

val_lin = k_fold_cross_val_mapes(10, train_data, labels, data_id, lin_model.ridge_normal_eq, cost, pred)

np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\LinRegr_Mapes.csv", val_lin, delimiter=",")

class model(torch_nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch_nn.Sequential(
            torch_nn.Linear(int(train_data.shape[1]), int(nn_opt[0])), 
            torch_nn.ReLU(), 
            torch_nn.Dropout(p = float(nn_opt[1])),
            torch_nn.Linear(int(nn_opt[0]), 1)
        )
        self.double()
    def forward(self, x):
        #x = torch_nn.Flatten(x) #Just reshapes to shape[0], shape[1] * ... * shape[n]
        return self.network(x)
def he_init(m):
    if isinstance(m, torch_nn.Linear):
        torch_nn.init.kaiming_normal_(m.weight)
        torch_nn.init.zeros_(m.bias)

nn = model()
nn.apply(he_init)
init_nn = deepcopy(nn.state_dict())

optimizer = optim.Adam(nn.parameters(), lr=float(nn_opt[3]), weight_decay=float(nn_opt[4]))
init_opt = deepcopy(optimizer.state_dict())

loss_fn = torch_nn.MSELoss(reduction='sum')

norm = [0, 0]

def nn_train(features, labels):
    nn.load_state_dict(init_nn)
    optimizer.load_state_dict(init_opt)

    std = np.std(features, axis = 0)
    std[std == 0] = 0.001
    norm[0], norm[1] = np.mean(features, axis = 0), std
    features_train = (features - norm[0]) / norm[1]
    dataset = TensorDataset(torch.from_numpy(features_train), torch.from_numpy(labels))
    dataloader = DataLoader(dataset, batch_size=int(nn_opt[2]), shuffle=True)
    for i in range(300):
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            y_batch_pred = nn(x_batch)
            loss = loss_fn(y_batch_pred, y_batch)
            loss.backward()
            optimizer.step()
def nn_cost(features, labels):
    features_test = (features - norm[0]) / norm[1]
    return torch.abs((torch.from_numpy(labels) - nn(torch.from_numpy(features_test))) / torch.from_numpy(labels)).detach().numpy()
def nn_pred(features):
    features_test = (features - norm[0]) / norm[1]
    return nn(torch.from_numpy(features_test)).detach().numpy()

val_nn = k_fold_cross_val_mapes(10, train_data, labels, data_id, nn_train, nn_cost, nn_pred)
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\NN_Mapes.csv", val_nn, delimiter=",")

xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
def train(features, labels):
    xgboost_reg.fit(features, labels)
def cost(features, labels):
    return np.abs(xgboost_reg.predict(features).reshape(labels.shape[0], 1) - labels) / labels
def pred_xgboost(features):
    return xgboost_reg.predict(features)
val_xgboost = k_fold_cross_val_mapes(10, train_data, labels, data_id, train, cost, pred_xgboost)
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\XGBoost_Mapes.csv", val_xgboost, delimiter=",")