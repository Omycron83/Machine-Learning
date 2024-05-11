import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.colors as mcolors

def mutual_information_table(df, discrete_columns):
    mi_table = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    for col1 in df.columns:
        for col2 in df.columns:
            if col2 in discrete_columns:
                mi_table.loc[col1, col2] = mutual_info_classif(df[[col1]], df[col2])
            else:
                mi_table.loc[col1, col2] = mutual_info_regression(df[[col1]], df[col2])[0]
    
    # Normalize the mutual information values
    scaler = MinMaxScaler()
    mi_table_normalized = pd.DataFrame(scaler.fit_transform(mi_table), index=mi_table.index, columns=mi_table.columns)
    return mi_table_normalized


data = pd.read_csv("D:\Damian\PC\Python\ML\Supervised_Learning\Projects\Brücken\DatensatzTraining_g_Test.csv")
data_id = data['ID_AUFTRAG_ZPM'].to_numpy()
data = data.drop(['ID_AUFTRAG_ZPM'], axis = 1) #
data = data.drop(['BETA_HT_Q_DEB'], axis = 1) 

discrete_columns = ['Ssp_Randkonstruktion', 'DEB_I_LAGERUNG_UEBERBAU2']


mi_table = mutual_information_table(data, discrete_columns)
mi_table.values[np.triu_indices_from(mi_table, k=1)] = np.nan

print("Mutual Information Table:")
print(mi_table)
colors = [(0.6, 0.6, 1), (0.1, 0.1, 0.5)]  # white to semi-dark blue
cmap = mcolors.LinearSegmentedColormap.from_list('CustomMap', colors, N=100)

plt.figure(figsize=(8, 6))
plt.imshow(mi_table, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='Mutual Information')
plt.title('Mutual Information Heatmap')
plt.xticks(range(len(mi_table.columns)), mi_table.columns, rotation=90)
plt.yticks(range(len(mi_table.index)), mi_table.index)

for i in range(len(mi_table.index)):
    for j in range(len(mi_table.columns)):
        plt.text(j, i, f'{mi_table.iloc[i, j]:.2f}', ha='center', va='center', color='white')

plt.tight_layout()
plt.show()


import numpy as np
import xgboost
import NN
from skopt import gp_minimize
from skopt.space import Real, Integer

labels = data["EIGENFREQ_ALT_STUFE_5"].to_numpy()
labels = labels.reshape((labels.shape[0], 1))
data = data.drop(["EIGENFREQ_ALT_STUFE_5"], axis = 1)
train_data = data.to_numpy()

def pca(data):
    data = (data - np.mean(data, axis = 0)) / (np.std(data, axis = 0) + (np.std(data, axis = 0) == 0)*0.001)
    cov = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    top_eigenvectors = eigenvectors[:, :8]
    return data @ top_eigenvectors

eigenvektoren = pca(train_data)

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
        return NN.mape(labels, pred)
    avg = 0
    for i in range(3):
        avg += k_fold_cross_val(10, eigenvektoren, labels, train, cost, seed = i) / 3
    return avg

xgboost_opt = gp_minimize(model_eval_xgboost, [Real(0, 20), Real(0.01, 0.5), Integer(3, 10), Integer(100, 1100), Real(0.5, 1), Integer(1, 10), Real(0, 5), Real(0, 5)], n_calls = 50)
print("XGBoost results:", "Optimum:", xgboost_opt.fun,"With values", xgboost_opt.x)