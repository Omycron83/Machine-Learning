#Final model parameters were:
lin_opt = [1.2776441026927368]
nn_opt = [555, 0.5119633527495826, 47, 0.0005015370459087969, 5.0]
xgboost_opt = [1.2373453011910522, 0.01, 6, 1096, 0.8544992034696949, 1, 0.8009271586748941, 1.4147922102986796]

import xgboost
import NN
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import LinRegr
import pandas as pd
import numpy as np

data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Va.csv")
data["BETA_HT_Q_DEB"] = data["BETA_HT_Q_DEB"].fillna(data["BETA_HT_Q_DEB"].mean())
labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1)).astype('float64')
train_data = data.to_numpy().astype('float64')
train_data = np.delete(train_data, 1, 1)

data_unknown = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_Va.csv")
data_unknown = data.to_numpy().astype('float64')
data_id = data_unknown[:, 0]
data_unknown = np.delete(data_unknown, 0, 1)

#Training the linear model with the optimal hyperparameters, visualizing
print("Starting visualization linear:")
lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
lin_model.ridge_normal_eq(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
figure, axis = plt.subplots(3, 1)
axis[0].scatter(labels[int(train_data.shape[0] * 0.8):], lin_model.predict(train_data[int(train_data.shape[0] * 0.8):]))
axis[0].set_title("Linear")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[0].transAxes
line.set_transform(transform)
axis[0].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[0].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[0].add_line(line)

#Training the nn_model, visualizing
print("Starting visualization nn:")
nn_model = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
nn_model.adam_iterated(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)], NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations = 1200)
nn_model.forward_propagation(train_data[int(train_data.shape[0] * 0.8):], labels[int(train_data.shape[0] * 0.8):], NN.MSE)
axis[1].scatter(labels[int(train_data.shape[0] * 0.8):], nn_model.output_layer())
axis[1].set_title("NN")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[1].transAxes
line.set_transform(transform)
axis[1].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[1].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[1].add_line(line)

#Training the xgboost model, visualizing
print("Starting visualization xgboost:")
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_model.fit(train_data[:int(train_data.shape[0] * 0.8)], labels[:int(train_data.shape[0] * 0.8)])
axis[2].scatter(labels[int(train_data.shape[0] * 0.8):], xgboost_model.predict(train_data[int(train_data.shape[0] * 0.8):]))
axis[2].set_title("XGBoost")
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = axis[2].transAxes
line.set_transform(transform)
axis[2].set_xlim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[2].set_ylim(0, data["EIGENFREQ_ALT_STUFE_5"].max())
axis[2].add_line(line)
plt.show()

#Saving the values:
print("Starting saving linear:")
lin_model = LinRegr.polynomial_regression(train_data.shape[1], _lambda = lin_opt[1], degree = lin_opt[0])
lin_model.ridge_normal_eq(train_data, labels)
pred_linregr = lin_model.predict(data_unknown)
np.savetxt("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\LinRegr.csv", np.vstack((data_id,  pred_linregr )), delimiter=",")

print("Starting saving nn:")
nn_model = NN.cont_feedforward_nn(train_data.shape[1], [nn_opt[0]], NN.ReLU, NN.ReLUDeriv, NN.output, NN.MSE_out_deriv, 1)
nn_model.adam_iterated(train_data, labels, NN.MSE, dropout= [nn_opt[1]], batchsize = nn_opt[2], alpha = nn_opt[3], _lambda = nn_opt[4], iterations = 250)
nn_model.forward_propagation(train_data, labels, NN.MSE)
pred_nn = nn_model.output_layer(data_unknown)
np.savetxt("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\NN.csv", np.vstack((data_id, pred_nn )), delimiter=",")

print("Starting saving xgboost:")
xgboost_model = xgboost.XGBRegressor(gamma = xgboost_opt[0], learning_rate = xgboost_opt[1], max_depth = xgboost_opt[2], n_estimators = xgboost_opt[3], n_jobs = 16, objective = 'reg:squarederror', subsample = xgboost_opt[4], scale_pos_weight = 0, reg_alpha = xgboost_opt[6], reg_lambda = xgboost_opt[7], min_child_weight = xgboost_opt[5])
xgboost_model.fit(train_data, labels)
pred_xgboost = xgboost_model.predict(data_unknown)
np.savetxt("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\XGBoost.csv", np.vstack((data_id, pred_xgboost )), delimiter=",")