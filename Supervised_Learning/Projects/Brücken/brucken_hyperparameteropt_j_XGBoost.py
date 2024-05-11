#Project that is supposed to be part of a paper exploring ML-applications in construction engineering
#Reason: for a certain type of bridge commonly found in the german railroad network, the current formula for calculating eigenfrequencies seems to systematically overestimate them compared to actual measurements done empirically 
#Through ML, a sharper estimate should be found based on the known features of the bridge as well as the result of the commonly used formula
#Here, we will use k-fold-crossvalidation in order to test multiple kinds of models for this application
#It is expected that XGBoost will produce superior performance on the tabular dataset, but we will explore multiple options

#Importing libraries
import warnings
#As there are a bunch of what have been found to be unenlightening warning messages those will be disabled for everyones convenience
warnings.filterwarnings("ignore")
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
now = datetime.now()
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#Loading in the data and preparing it
data = pd.read_csv(r"DatensatzTraining_j.csv")
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1)
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

#Implementing k-fold-cv in order to efficiently perform hyperparameter search on the limited data available 
global n
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

#The hyperparameter tuning for each kind of model will be done using bayesian hyperparameter-optimization as implemented in the scikit-optimize library 
#For the library to work, we only need the hyperparameter search space as well as the objective function (just a specification of the code above)
#This is done for all models by keeping the parameters as well as the other functions variable
#The parameters are represented as tuples, which are then unpacked in the function call (with "*")
def mape_obj(preds, dtrain):
    labels = dtrain
    errors = np.abs((labels - preds) / labels) / preds.shape[0]
    grad = np.sign(labels - preds) / labels / preds.shape[0]
    hess = np.zeros_like(grad)
    return grad, hess

#For XGBoost:
#params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda
n = 0
def model_eval_xgboost(params):
    global n
    n += 1
    if n % 1 == 0:
        print("Iteration:", n)
    xgboost_reg = xgboost.XGBRegressor(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = mape_obj, subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])
    def train(features, labels):
        xgboost_reg.fit(features, labels)
    def cost(features, labels):
        pred = xgboost_reg.predict(features).reshape(labels.shape[0], 1)
        return NN.mape(labels, pred)
    avg = 0
    for i in range(3):
        avg += k_fold_cross_val(10, train_data, labels, train, cost, seed = i) / 3
    return avg

xgboost_opt = gp_minimize(model_eval_xgboost, [Real(0, 20), Real(0.01, 0.5), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], n_calls = 25)
print("XGBoost results:", "Optimum:", xgboost_opt.fun,"With values", xgboost_opt.x)
file_xgboost = open(r"XGBoost.txt", "a")
now = datetime.now()
file_xgboost.write("\n" + str(now) + ":" + str(xgboost_opt.fun) + " " + str(xgboost_opt.x) + " ")
file_xgboost.close()
"""
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt.x[0], learning_rate = xgboost_opt.x[1], max_depth = xgboost_opt.x[2], n_estimators = xgboost_opt.x[3], n_jobs = 16, objective = 'reg:mape', subsample = xgboost_opt.x[4], scale_pos_weight = 0, reg_alpha = xgboost_opt.x[6], reg_lambda = xgboost_opt.x[7], min_child_weight = xgboost_opt.x[5])
xgboost_model.fit(train_data[:int(train_data.shape[0] * 0.7)], labels[:int(train_data.shape[0] * 0.7)])
axis[2].scatter(labels[int(train_data.shape[0] * 0.7):], xgboost_model.predict(train_data[int(train_data.shape[0] * 0.7):]))
axis[2].set_title("XGBoost")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[2].transAxes
line.set_transform(transform)
axis[2].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[2].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[2].add_line(line)
plt.show()
"""