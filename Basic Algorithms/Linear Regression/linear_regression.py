import numpy as np
class linear_regression:
    def __init__(self, dim, _lambda = 0) -> None:
        self.theta = np.zeros((dim + 1, 1))
        self._lambda = 0
    def normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.inv(features_bias.T @ features_bias)@ features_bias.T@labels

    def ridge_normal_eq(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        self.theta = np.linalg.inv(features_bias.T @ features_bias + self._lambda * np.identity(features_bias.shape[1])) @ features_bias.T@labels
    
    def MSE(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        return np.sum((features_bias @ self.theta - labels)**2) / features.shape[0]