import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import NN
import xgboost
import LinRegr
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from datetime import datetime
import torch
from torch import nn as torch_nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

xgboost_opt = [0.0, 0.07896507820168863, 6, 969, 0.7184912525930043, 2, 0.0, 3.211285854810482]

data = pd.read_csv(r"DatensatzTraining_i.csv")
data = data[~data['ID_AUFTRAG_ZPM'].isin([5709,5708,50501,8514,28216])]
data_id = data['ID_AUFTRAG_ZPM'].to_numpy().reshape(-1, 1)
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])

def k_fold_cross_val_mapes(k, features, labels, indices, train_func, cost_func, seed = 0):
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
        error += [np.hstack((test_indices.reshape(-1, 1), cost_func(test_features, test_labels).reshape(-1, 1)))]

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
    error += [np.hstack((test_indices.reshape(-1, 1), cost_func(test_features, test_labels).reshape(-1, 1)))]
    return np.vstack(error)

xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
def train(features, labels):
    xgboost_reg.fit(features, labels)
def cost(features, labels):
    return np.abs(xgboost_reg.predict(features).reshape(labels.shape[0], 1) - labels) / labels
def pred(features):
    return xgboost_reg.predict(features) - 
val_xgboost = k_fold_cross_val_mapes(10, train_data, labels, data_id, train, cost)

