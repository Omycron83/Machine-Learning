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
        reg_matrix = np.identity(features_bias.shape[1])
        #We need to make sure to not regularize the bias weight, which is done by setting its entry to 0
        reg_matrix[features_bias.shape[1] - 1, features_bias.shape[1] - 1] = 0
        self.theta = np.linalg.inv(features_bias.T @ features_bias + self._lambda * reg_matrix) @ features_bias.T @labels

    def predict(self, features):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        return features_bias @ self.theta
    
    def MSE(self, features, labels):
        features_bias = np.hstack((features, np.ones((features.shape[0], 1))))
        pred = (features_bias @ self.theta).reshape(features_bias.shape[0], self.theta.shape[1])
        return np.sum((np.array(pred) - np.array(labels).reshape(pred.shape))**2) / (2 * np.array(pred).size)
    
class polynomial_regression(linear_regression):
    def __init__(self, dim, dim_2=1, _lambda=0, degree = 1) -> None:
        self.degree = degree
        super().__init__(dim * degree, dim_2, _lambda)

    def polynomialize(self, features):
        return np.hstack(features ** np.arange(1, self.degree + 1)[:, None, None])
    
    def normal_eq(self, features, labels):
        return super().normal_eq(self.polynomialize(features), labels)
    
    def ridge_normal_eq(self, features, labels):
        return super().ridge_normal_eq(self.polynomialize(features), labels)
    
    def MSE(self, features, labels):
        return super().MSE(self.polynomialize(features), labels)

    def predict(self, features):
        return super().predict(self.polynomialize(features))
"""
x = np.arange(9).reshape(3,3)
print(x)
y = np.arange(3)

g = polynomial_regression(3, 1, degree=12, _lambda = 0)
g.ridge_normal_eq(x, y)
print(g.predict(x), y)
"""