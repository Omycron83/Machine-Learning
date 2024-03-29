#Final model parameters were:
lin_opt = [1, 36.66342272433806]
nn_opt = [128, 0, 18, 0.00001, 0]
xgboost_opt = [0.0, 0.031216969731852354, 8, 837, 1.0, 3, 0.0, 1.72237750519606]

import xgboost
import NN
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import LinRegr
import pandas as pd
import numpy as np

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

nn = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
untrained_weights = nn.retrieve_weights()
def train(features, labels):
    nn.assign_weights(untrained_weights)
    nn.adam_iterated(features, labels, NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations = 300)
def cost(features, labels):
    nn.forward_propagation(features, labels, NN.MSE)
    return NN.mape(labels, nn.output_layer())
avg = 0
for i in range(3):
    val = k_fold_cross_val(10, train_data, labels, train, cost, seed = i)
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
nn_model = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
nn_model.adam_iterated(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)], NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations = 300)
nn_model.forward_propagation(train_data[int(train_data.shape[0] * 0.8):], labels[int(train_data.shape[0] * 0.8):], NN.MSE)
print(NN.mape(labels[int(train_data.shape[0] * 0.8):],nn_model.output_layer()))
axis[1].scatter(labels[int(train_data.shape[0] * 0.8):], nn_model.output_layer())
axis[1].set_title("NN")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[1].transAxes
line.set_transform(transform)
axis[1].set_xlim(0, labels.max())
axis[1].set_ylim(0, labels.max())
axis[1].add_line(line)

#Training the xgboost model, visualizing
print("Starting visualization xgboost:")
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_model.fit(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
print(NN.mape(labels[int(train_data.shape[0] * 0.8):],xgboost_model.predict(train_data[int(train_data.shape[0] * 0.8):])))
axis[2].scatter(labels[int(train_data.shape[0] * 0.8):], xgboost_model.predict(train_data[int(train_data.shape[0] * 0.8):]))
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
pred_linregr = lin_model.predict(data_unknown)
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\LinRegr.csv", np.hstack((data_id.reshape(-1, 1),  pred_linregr, data_stw.reshape(-1, 1))), delimiter=",")

print("Starting saving nn:")
nn_model = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
nn_model.adam_iterated(train_data, labels, NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations = 300)
nn_model.forward_propagation(data_unknown, np.zeros((data_unknown.shape[0], 1)), NN.MSE)
pred_nn = nn_model.output_layer()
plt.scatter(np.log(data_stw), pred_nn)
plt.show()
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\NN.csv", np.hstack((data_id.reshape(-1, 1), pred_nn, data_stw.reshape(-1, 1))), delimiter=",")

print("Starting saving xgboost:")
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_model.fit(train_data, labels)
pred_xgboost = xgboost_model.predict(data_unknown)
np.savetxt(r"D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\XGBoost.csv", np.hstack((data_id.reshape(-1, 1), pred_xgboost.reshape(-1, 1), data_stw.reshape(-1, 1))), delimiter=",")