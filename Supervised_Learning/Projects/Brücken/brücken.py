#Project that is supposed to be part of a paper exploring ML-applications in construction engineering
#Reason: for a certain type of bridge commonly found in the german railroad network, the current formula for calculating eigenfrequencies seems to systematically overestimate them compared to actual measurements done empirically 
#Through ML, a sharper estimate should be found based on the known features of the bridge as well as the result of the commonly used formula
#Here, we will use k-fold-crossvalidation in order to test multiple kinds of models for this application
#It is expected that XGBoost will produce superior performance on the tabular dataset, but we will explore multiple options

#Importing libraries
import warnings
#As there are a bunch of what have been found to be unenlightening warning messages those will be disabled for everyones convenience
warnings.filterwarnings("ignore")

import NN
import xgboost
import LinRegr 
import pandas as pd
import numpy as np
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer

#Loading in the data and preparing it
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Va.csv")
data["BETA_HT_Q_DEB"] = data["BETA_HT_Q_DEB"].fillna(data["BETA_HT_Q_DEB"].mean())
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
train_data = data.to_numpy()
train_data = np.delete(train_data, 1, 1)

#Implementing k-fold-cv in order to efficiently perform hyperparameter search on the limited data available 
def k_fold_cross_val(k, features, labels, train_func, cost_func):
    #Shuffling
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

#For the linear regression (with L2-Penalization):
#params[0] = lambda
def model_eval_linear(params):
    lin_model = LinRegr.linear_regression(train_data.shape[1], _lambda = params[0])
    return k_fold_cross_val(10, train_data, labels, lin_model.ridge_normal_eq, lin_model.MSE)
lin_opt = gp_minimize(model_eval_linear, [Real(0, 5)], n_calls = 10)
print(lin_opt.fun, lin_opt.x)
#print(lin_opt.func_vals)
#print(lin_opt.x_iters)
#lin_test = LinRegr.linear_regression(train_data.shape[1], 0)
#lin_test.ridge_normal_eq(train_data, labels)
#print(lin_test.MSE(train_data, labels))

#For the neural network:
#params[0] = neurons, params[1] = dropout_perc, params[2] = batchsize adam, params[3] = learning rate, params[4] = lambda 
def model_eval_nn(params):
    nn = NN.cont_feedforward_nn(train_data.shape[1], [params[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
    untrained_weights = nn.retrieve_weights()
    def train(features, labels):
        nn.assign_weights(untrained_weights)
        last = np.finfo(np.float64).max
        curr = nn.adam(features, labels, NN.MSE, dropout= [params[1]], batchsize = params[2], alpha = params[3], _lambda = params[4])
        for i in range(150):
            if last*0.998 < curr:
                break
            last = curr
            curr = nn.adam(features, labels, NN.MSE, dropout= [params[1]], batchsize = params[2], alpha = params[3], _lambda = params[4])

    def cost(features, labels):
        return nn.forward_propagation(features, labels, NN.MSE)
    
    return k_fold_cross_val(10, train_data, labels, train, cost)
nn_opt = gp_minimize(model_eval_nn, [Integer(1, 1024), Real(0, 0.9999), Integer(8, 128), Real(0.00001, 0.001), Real(0, 5)], n_calls = 10)
print(nn_opt.fun, nn_opt.x)

#For XGBoost:
#params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda
def model_eval_xgboost(params):
    xgboost_reg = xgboost.XGBRegressor(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'reg:squarederror', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])
    def train(features, labels):
        xgboost_reg.fit(features, labels)
    def cost(features, labels):
        pred = xgboost_reg.predict(features).reshape(labels.shape[0], 1)
        return NN.MSE(pred, labels)
    return k_fold_cross_val(10, train_data, labels, train, cost)
xgboost_opt = gp_minimize(model_eval_xgboost, [Real(0, 20), Real(0.01, 0.2), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], n_calls = 10)
print(xgboost_opt.fun, xgboost_opt.x)


