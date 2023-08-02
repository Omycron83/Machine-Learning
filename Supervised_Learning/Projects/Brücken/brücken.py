#Project that is supposed to be part of a paper exploring ML-applications in construction engineering
#Reason: for a certain type of bridge commonly found in the german railroad network, the current formula for calculating eigenfrequencies seems to systematically overestimate them compared to actual measurements done empirically 
#Through ML, a sharper estimate should be found based on the known features of the bridge as well as the result of the commonly used formula
#Here, we will use k-fold-crossvalidation in order to test multiple kinds of models for this application
#It is expected that XGBoost will produce superior performance on the tabular dataset, but we will explore multiple options

#Importing libraries
import NN
import xgboost
import linear_regression
import pandas as pd
import numpy as np
from skopt.utils import use_named_args
from skopt import gp_minimize

#Loading in the data and preparing it
data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken_NN\DatensatzTraining.csv")
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
train_data = data.to_numpy()
train_data = np.delete(train_data, 1, 1)

#Implementing k-fold-cv in order to efficiently perform hyperparameter search on the limited data available 
#This is done for all models by keeping the parameters as well as the other functions variable
#The parameters are represented as tuples, which are then unpacked in the function call (with "*")
def k_fold_cross_val(k, features, labels, parameters, train_func, cost_func):
    #Shuffling
    p = np.random.permutation(features.shape[0])
    shuffled_features = features[p]
    shuffled_labels = labels[p]
    error = 0
    for l in range(k - 1):
        #The test data of the current fold
        test_features = shuffled_features[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        test_labels = shuffled_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        #The remaining training data of the current fold
        train_features = shuffled_features[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        train_labels = shuffled_labels[data.shape[0] // k * l:data.shape[0] // k * (l+1), :]
        #Now, train the model on the current fold
        train_func(train_features, train_labels, *parameters)
        error += cost_func(test_features, test_labels) / k

    #For the last fold, we dont really know the size of the holdout-set (we dont know about the divisibility of the amount of datapoints by k) so we do this seperately
    #The test data of the last fold 
    test_features = shuffled_features[0 : data.shape[0] // k * (l + 1), :]
    test_labels = shuffled_labels[0 : data.shape[0] // k * (l + 1), :]
    #The remaining training data of the last fold
    train_features = shuffled_features[data.shape[0] // k * (l + 1):, :]
    train_labels = shuffled_labels[data.shape[0] // k * (l + 1):, :]
    #Now, train the model on the current fold
    train_func(train_features, train_labels, *parameters)
    error += cost_func(test_features, test_labels) / k
    return error

#The hyperparameter tuning for each kind of model will be done using bayesian hyperparameter-optimization as implemented in the scikit-optimize library 
#For the library to work, we only need the hyperparameter search space as well as the objective function (just a specification of the code above)

#For the linear regression (with L2-Penalization):

search_space = [(0, 5)]
@use_named_args(search_space)
def model_eval_linear(*params):
    lin_model = new 

#For the neural network:

#For XGBoost:




