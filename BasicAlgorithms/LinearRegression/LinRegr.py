import numpy as np
class linear_regression:
    def __init__(self, dim, dim_2 = 1, _lambda = 0) -> None:
        self.theta = np.zeros((dim + 1, dim_2))
        self._lambda = _lambda
    
    def normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.pinv(features_bias.T @ features_bias)@ features_bias.T@labels

    def ridge_normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.inv(features_bias.T @ features_bias + self._lambda * np.identity(features_bias.shape[1])) @ features_bias.T @labels

    def predict(self, features):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        return features_bias @ self.theta
    def MSE(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        pred = (features_bias @ self.theta).reshape(features_bias.shape[0], self.theta.shape[1])
        return np.sum((np.array(pred) - np.array(labels).reshape(pred.shape))**2) / (2 * np.array(pred).size)