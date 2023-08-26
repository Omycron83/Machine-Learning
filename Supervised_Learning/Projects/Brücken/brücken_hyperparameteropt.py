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

#For the polynomial regression (with L2-Penalization):
#params[0] = degree, params[1] = lambda
n = 0
def model_eval_linear(params):
    global n
    n += 1
    if n % 100 == 0:
        print("Iteration:", n)
    lin_model = LinRegr.polynomial_regression(train_data.shape[1], degree = params[0], _lambda = params[1])
    avg = 0
    for i in range(3):
        avg += k_fold_cross_val(10, train_data, labels, lin_model.ridge_normal_eq, lin_model.MSE, seed = i) / 3
    return avg
lin_opt = gp_minimize(model_eval_linear, [Integer(1, 20), Real(0, 40)], n_calls = 1200)
#Printing out the top results
print("Linear results:", "Optimum:", lin_opt.fun,"With values", lin_opt.x)
file_linregr = open("LinRegr.txt", "a")
file_linregr.write(str(lin_opt.fun) + " " + str(lin_opt.x))
file_linregr.close()
#Regressing on 70% of the data and then plotting the output variables on the test set vs real values
lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt.x[1], degree = lin_opt.x[0])
lin_model.ridge_normal_eq(train_data[:int(train_data.shape[0] * 0.7)], labels[:int(train_data.shape[0] * 0.7)])
figure, axis = plt.subplots(3, 1)
axis[0].scatter(labels[int(train_data.shape[0] * 0.7):], lin_model.predict(train_data[int(train_data.shape[0] * 0.7):]))
axis[0].set_title("Linear")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[0].transAxes
line.set_transform(transform)
axis[0].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[0].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[0].add_line(line)

#For the neural network:
#params[0] = neurons, params[1] = dropout_perc, params[2] = batchsize adam, params[3] = learning rate, params[4] = lambda 
n = 0
def model_eval_nn(params):
    global n
    n += 1
    if n % 10 == 0:
        print("Iteration:", n)
    nn = NN.cont_feedforward_nn(train_data.shape[1], [params[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
    untrained_weights = nn.retrieve_weights()
    def train(features, labels):
        nn.assign_weights(untrained_weights)
        nn.early_stopping_adam_iterated(features, labels, NN.MSE, dropout= [params[1]], batchsize = params[2], alpha = params[3], _lambda = params[4], iterations = 300)
    def cost(features, labels):
        return nn.forward_propagation(features, labels, NN.MSE)
    avg = 0
    for i in range(3):
        avg += k_fold_cross_val(10, train_data, labels, train, cost, seed = i) / 3
    return avg

nn_opt = gp_minimize(model_eval_nn, [Integer(1, 2048), Real(0, 0.9999), Integer(8, 128), Real(0.0001, 0.01), Real(0, 20)], n_calls = 1200)
print("NN results:", "Optimum:", nn_opt.fun,"With values", nn_opt.x)
file_NN = open("NN.txt", "a")
file_NN.write(str(nn_opt.fun) + " " + str(nn_opt.x))
file_NN.close()
nn_model = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt.x[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
for i in range(250):
    l = nn_model.adam(train_data[:int(train_data.shape[0] * 0.7)], labels[:int(train_data.shape[0] * 0.7)], NN.MSE, dropout= [nn_opt.x[1]], batchsize = nn_opt.x[2], alpha = nn_opt.x[3], _lambda = nn_opt.x[4])
    if i % 10 == 0:
        print(l)
nn_model.forward_propagation(train_data[int(train_data.shape[0] * 0.7):], labels[int(train_data.shape[0] * 0.7):], NN.MSE)
axis[1].scatter(labels[int(train_data.shape[0] * 0.7):], nn_model.output_layer())
axis[1].set_title("NN")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[1].transAxes
line.set_transform(transform)
axis[1].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[1].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[1].add_line(line)
#For XGBoost:
#params[0] = gamma, params[1] = learning_rate, params[2] = max_depth, params[3] = n_estimators, params[4] = sub_sample, params[5] = min_child_weight, params[6] = reg_alpha, params[7] = reg_lambda
n = 0
def model_eval_xgboost(params):
    global n
    n += 1
    if n % 100 == 0:
        print("Iteration:", n)
    xgboost_reg = xgboost.XGBRegressor(gamma = params[0], learning_rate = params[1], max_depth = params[2], n_estimators = params[3], n_jobs = 16, objective = 'reg:squarederror', subsample = params[4], scale_pos_weight = 0, reg_alpha = params[6], reg_lambda = params[7], min_child_weight = params[5])
    def train(features, labels):
        xgboost_reg.fit(features, labels)
    def cost(features, labels):
        pred = xgboost_reg.predict(features).reshape(labels.shape[0], 1)
        return NN.MSE(pred, labels)
    avg = 0
    for i in range(3):
        avg += k_fold_cross_val(10, train_data, labels, train, cost, seed = i) / 3
    return avg

xgboost_opt = gp_minimize(model_eval_xgboost, [Real(0, 20), Real(0.01, 0.5), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], n_calls = 1200)
print("XGBoost results:", "Optimum:", xgboost_opt.fun,"With values", xgboost_opt.x)
file_xgboost = open("xgboost.txt", "a")
file_xgboost.write(str(xgboost_opt.fun) + " " + str(xgboost_opt.x))
file_xgboost.close()
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt.x[0], learning_rate = xgboost_opt.x[1], max_depth = xgboost_opt.x[2], n_estimators = xgboost_opt.x[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt.x[4], scale_pos_weight = 0, reg_alpha = xgboost_opt.x[6], reg_lambda = xgboost_opt.x[7], min_child_weight = xgboost_opt.x[5])
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