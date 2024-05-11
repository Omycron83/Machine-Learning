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
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

test_data = pd.read_csv(r"DatensatzTesting_i.csv")
data_id = test_data['ID_AUFTRAG_ZPM'].to_numpy().reshape(-1, 1)
test_data.loc[test_data['ID_AUFTRAG_ZPM'] == 11531, 'Ssp_Randkonstruktion'] = 0
test_data = test_data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
test_labels = test_data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
test_labels = test_labels.reshape((test_labels.shape[0], 1))
test_data = test_data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
test__data = test_data.to_numpy()

xgboost_reg = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_reg.fit(train_data, labels)

def cost(features, labels):
    return np.abs(xgboost_reg.predict(features).reshape(labels.shape[0], 1) - labels) / labels

print(cost(test__data, test_labels), np.mean(cost(test__data, test_labels)))
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\XGBoost_Mapes_Test.csv", np.hstack((data_id, cost(test__data, test_labels), xgboost_reg.predict(test__data).reshape(test_labels.shape[0], 1))), delimiter=",")
print("HI")