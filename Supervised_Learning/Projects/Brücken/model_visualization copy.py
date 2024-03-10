#Final model parameters were:
lin_opt = [3, 0.03146046251720122]
nn_opt = [1668, 0.0, 33, 0.002580686067199954, 0.0]
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
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Ve.csv")
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
data["BETA_HT_Q_DEB"] = data["BETA_HT_Q_DEB"].fillna(data["BETA_HT_Q_DEB"].mean())
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

data_unknown = pd.read_csv(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\data_deployment_Modell_Ve.csv")
data_unknown = data_unknown.to_numpy()
data_id = data_unknown[:, 0]
data_stw = data_unknown[:, 2]
data_unknown = np.delete(data_unknown, 0, 1)

#Getting the final models score:
def k_fold_cross_val(k, features, labels, train_func, cost_func, seed):
    #Shuffling
    np.random.seed(seed)
    p = np.random.permutation(features.shape[0])
    shuffled_features = features.copy()[p]
    shuffled_labels = labels.copy()[p]
    error = 0
    for l in range(k - 1):
        #The test data of the current fold
        test_features = shuffled_features[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        test_labels = shuffled_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        #The remaining training data of the current fold
        train_features = np.vstack((shuffled_features[:data.shape[0] // k * l, :], shuffled_features[data.shape[0] // k * (l+1):, :]))
        train_labels = np.vstack((shuffled_labels[:data.shape[0] // k * l, :], shuffled_labels[data.shape[0] // k * (l+1):, :])) 
        #Now, train the model on the current fold
        train_func(train_features, train_labels)
        error += cost_func(test_features, test_labels) / k

    #For the last fold, we dont really know the size of the holdout-set (we dont know about the divisibility of the amount of datapoints by k) so we do this seperately
    #The test data of the last fold 
    test_features = shuffled_features[data.shape[0] // k * (l + 1):, :]
    test_labels = shuffled_labels[data.shape[0] // k * (l + 1):, :]
    #The remaining training data of the last fold
    train_features = shuffled_features[:data.shape[0] // k * (l + 1), :]
    train_labels = shuffled_labels[:data.shape[0] // k * (l + 1), :]
    #Now, train the model on the current fold
    train_func(train_features, train_labels)
    error += cost_func(test_features, test_labels) / k
    return error

lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
def cost(features, labels):
    return NN.mape(labels, lin_model.predict(features))
avg = 0
for i in range(3):
    val = k_fold_cross_val(10, train_data, labels, lin_model.ridge_normal_eq, cost, seed = i)
    print(val)
    avg += val / 3
print("Avg Lin:", avg)
print("-----------")

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
    return torch.mean(torch.abs((torch.from_numpy(labels) - nn(torch.from_numpy(features_test))) / torch.from_numpy(labels)))

avg = 0
for i in range(3):
    val = k_fold_cross_val(10, train_data, labels, nn_train, nn_cost, seed = i)
    print(val)
    avg += val / 3
print("Avg NN: ",avg)
print("-----------")

xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
def train(features, labels):
    xgboost_reg.fit(features, labels)
def cost(features, labels):
    pred = xgboost_reg.predict(features).reshape(labels.shape[0], 1)
    return NN.mape(labels, pred)
avg = 0
for i in range(3):
    val = k_fold_cross_val(10, train_data, labels, train, cost, seed = i)
    print(val)
    avg += val / 3
print("Avg XGBoost:",avg)
print("-----------")

#Training the linear model with the optimal hyperparameters, visualizing
print("Starting visualization linear:")
lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
lin_model.ridge_normal_eq(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
print(NN.mape(labels[int(train_data.shape[0] * 0.8):],lin_model.predict(train_data[int(train_data.shape[0] * 0.8):])))
figure, axis = plt.subplots(3, 1)
axis[0].scatter(labels[int(train_data.shape[0] * 0.8):], lin_model.predict(train_data[int(train_data.shape[0] * 0.8):]))
axis[0].set_title("Linear")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[0].transAxes
line.set_transform(transform)
axis[0].set_xlim(0, labels.max())
axis[0].set_ylim(0, labels.max())
axis[0].add_line(line)

#Training the nn_model, visualizing
print("Starting visualization nn:")
nn_train(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
print(nn_cost(train_data[int(train_data.shape[0] * 0.8):], labels[int(train_data.shape[0] * 0.8):]))
axis[1].scatter(labels[int(train_data.shape[0] * 0.8):], nn(torch.from_numpy((train_data[int(train_data.shape[0] * 0.8):] - norm[0]) / norm[1])).detach().numpy())
axis[1].set_title("NN")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[1].transAxes
line.set_transform(transform)
axis[1].set_xlim(0, labels.max())
axis[1].set_ylim(0, labels.max())
axis[1].add_line(line)

#Training the xgboost model, visualizing
print("Starting visualization xgboost:")
train(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
print(cost(train_data[int(train_data.shape[0] * 0.8):], labels[int(train_data.shape[0] * 0.8):]))
axis[2].scatter(labels[int(train_data.shape[0] * 0.8):], xgboost_reg.predict(train_data[int(train_data.shape[0] * 0.8):]))
axis[2].set_title("XGBoost")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[2].transAxes
line.set_transform(transform)
axis[2].set_xlim(0, labels.max())
axis[2].set_ylim(0, labels.max())
axis[2].add_line(line)
plt.show()


#Saving the predicted values for unknown data:
print("Starting saving linear:")
lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
lin_model.ridge_normal_eq(train_data, labels)
print(NN.mape(lin_model.predict(train_data), labels))
pred_linregr = lin_model.predict(data_unknown)
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\LinRegr.csv", np.hstack((data_id.reshape(-1, 1),  pred_linregr, data_stw.reshape(-1, 1))), delimiter=",")

print("Starting saving nn:")
nn_train(train_data, labels)
pred_nn = nn(torch.from_numpy((data_unknown - norm[0]) / norm[1])).detach().numpy()
print(NN.mape(nn(torch.from_numpy((train_data - norm[0]) / norm[1])).detach().numpy(), labels))
plt.scatter(data_stw, pred_nn)
plt.show()
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\NN.csv", np.hstack((data_id.reshape(-1, 1), pred_nn, data_stw.reshape(-1, 1))), delimiter=",")

print("Starting saving xgboost:")
train(train_data, labels)
pred_xgboost = xgboost_reg.predict(data_unknown)
print(NN.mape(xgboost_reg.predict(train_data), labels))
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\XGBoost.csv", np.hstack((data_id.reshape(-1, 1), pred_xgboost.reshape(-1, 1), data_stw.reshape(-1, 1))), delimiter=",")