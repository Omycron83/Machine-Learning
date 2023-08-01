import numpy as np

#This is my implementation of decision trees, random forests and XGBoost
#Decision trees are meant to be strong learners on their own, with instable trees being able to form a more stable ensemble, a random forst
#Gradient boosted trees on the other hand use weak lerners and do their own stuff
#data = Nested array, 1st dim datapoint, 2nd dim: Nr, x1, x2, ... ,xn , y 
class decision_tree:
    class node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
            self.predicts = None
        def toString(self, level=0):
            if self.predicts != None:
                ret = "\t"*level+str(self.val)+ " " +str(self.predicts)+"\n"
            else:
                ret = "\t"*level+str(self.val)+"\n"
            if self.left != None:
                ret += self.left.toString(level+1)
            if self.right != None:
                ret += self.right.toString(level+1)
            return ret

    def construct_tree(self, depth, max_depth, data, allowed_features):
        if len(data) == 0:
            curr_split = self.node("Final/No Data")
            curr_split.predicts = "Whats up"
            return curr_split
        if depth >= max_depth:
            percent_one = 0
            for i in range (len(data)):
                percent_one += data[i][-1]
            percent_one = (percent_one / len(data)) >= 0.5
            curr_split = self.node("Final/Max Depth")
            curr_split.predicts = percent_one
            return curr_split
        curr_split, split1, split2, entropy = self.get_best_split(data, allowed_features)
        if entropy == 0:
            percent_one = 0
            for i in range (len(data)):
                percent_one += data[i][-1]
            percent_one = (percent_one / len(data)) >= 0.5
            curr_split = self.node("Final/Good Split")
            curr_split.predicts = percent_one
            return curr_split
        else:
            curr_split = self.node(curr_split)
            curr_split.left = self.construct_tree(depth + 1, max_depth, split1, allowed_features)
            curr_split.right = self.construct_tree(depth + 1, max_depth, split2, allowed_features)
            return curr_split

    def predict(self, data):
        predicted = self.parse_tree(data, self.tree)
        predicted.sort(key= lambda x: x[0])
        return predicted
        
    def __str__(self) -> str:
        return self.tree.toString()

    def __init__(self, data, max_depth, allowed_features = []):
        if len(allowed_features) == 0:
            allowed_features = [i for i in range(1, len(data[0]) - 1)] 
        self.max_depth = max_depth
        self.data = data
        #The tree nodes are arrays with 2 values: the value that was split at and the feature index affected
        self.tree = self.construct_tree(0, max_depth, data, allowed_features)

class regression_tree(decision_tree):
    def parse_tree(self, data, node):
        pred = []
        if node.predicts == None:
            entropy, split1, split2 = self.construct_split_regression(data, node.val[1], node.val[0])
            if self.parse_tree(split1, node.left) != None:
                pred += self.parse_tree(split1, node.left)
            if self.parse_tree(split2, node.right) != None:
                pred += self.parse_tree(split2, node.right)
            return pred
        else:
            data_list = []
            for i in range(len(data)):
                new_data = data[i].copy()
                new_data[-1] = int(node.predicts)
                data_list.append(new_data)
            data_list.sort(key= lambda x: x[0])
            return data_list
    
    def get_best_split(self, data, features):
        data_2 = []
        for i in range(len(data)):
            data_2.append(data[i][-1])
        curr_entropy = self.get_variance(data_2)
        min_entropy = 10000000
        curr_split = [data[0][1], 1]
        for i in range(1, len(data[0]) - 1):
            if i in features:
                for j in range(len(data)):
                    entropy, split1, split2 = self.construct_split_regression(data, i, data[j][i])
                    if entropy < min_entropy:
                        min_entropy = entropy
                        curr_split = [data[j][i], i]
                        curr_split1 = split1
                        curr_split2 = split2
        if min_entropy < curr_entropy:
            return curr_split, curr_split1, curr_split2, entropy
        else:
            return curr_split, data, [], 0
    
    def construct_split_regression(self, data, index, value):
        split1, split2 = [], []
        for i in range(len(data)):
            if data[i][index] >= value:
                split1.append(data[i])
            else:
                split2.append(data[i])
        split1_y = []
        for i in range(len(split1)):
            split1_y.append(split1[i][-1])
        split2_y = []
        for i in range(len(split2)):
            split2_y.append(split2[i][-1])
        return self.get_variance(split1_y) * len(split1) / len(data) + self.get_variance(split2_y) * len(split2) / len(data), split1, split2
    
    def get_variance(self, data):
        if len(data) > 1:
            var = 0
            mean = np.mean(data)
            for i in data:
                var += (mean - i)**2 / (len(data)-1)
        else:
            var = 0
        return var
    def MSE(self, y_hat, data):
        if len(data) > 0:
            MSE = 0
            for i in range(len(data)):
                MSE += (data[i][-1] - y_hat[i][-1])**2
            MSE = MSE / len(data)
            return MSE
        else:
            return 0
    def get_error(self, y_hat, data):
        return self.MSE(y_hat, data)
    

class classification_tree(decision_tree):
    def get_best_split(self, data, features):
        curr_entropy = 0
        for i in range(len(data)):
            curr_entropy += data[i][-1] / len(data)
        curr_entropy = self.get_entropy(curr_entropy)
        min_entropy = 10000000
        curr_split = [data[0][features[0]], features[0]]
        for i in range(1, len(data[0]) - 1):
            if i in features:
                for j in range(len(data)):
                    entropy, split1, split2 = self.construct_split_classification(data, i, data[j][i])
                    if entropy < min_entropy:
                        min_entropy = entropy
                        curr_split = [data[j][i], i]
                        curr_split1 = split1
                        curr_split2 = split2
        if min_entropy < curr_entropy:
            return curr_split, curr_split1, curr_split2, entropy
        else:
            return curr_split, curr_split1, curr_split2, 0
    def parse_tree(self, data, node):
        pred = []
        if node.predicts == None:
            entropy, split1, split2 = self.construct_split_classification(data, node.val[1], node.val[0])
            if self.parse_tree(split1, node.left) != None:
                pred += self.parse_tree(split1, node.left)
            if self.parse_tree(split2, node.right) != None:
                pred += self.parse_tree(split2, node.right)
            return pred
        else:
            data_list = []
            for i in range(len(data)):
                new_data = data[i].copy()
                new_data[-1] = node.predicts
                data_list.append(new_data)
            return data_list
    
    def construct_split_classification(self, data, index, value):
        entropy1, entropy1_counter = 0, 0
        entropy2, entropy2_counter = 0, 0
        split1, split2 = [], []
        for i in range(len(data)):
            if data[i][index] >= value:
                split1.append(data[i])
                entropy1 += (data[i][-1]==1)
                entropy1_counter += 1
            else:
                split2.append(data[i])
                entropy2 += (data[i][-1]==1)
                entropy2_counter += 1
        if entropy1_counter != 0:
            entropy1 /= entropy1_counter
        if entropy2_counter != 0:
            entropy2 /= entropy2_counter
        return self.get_entropy(entropy1) * entropy1_counter / len(data) + self.get_entropy(entropy2) * entropy2_counter / len(data), split1, split2
    
    def get_entropy(self, p1):
        return -p1*np.log2(p1 + int(p1 == 0) * 0.0001) - (1-p1)*np.log2(1-p1 + int(p1 == 1) * 0.00001)
    
    def logistic_cost(self, y_hat, y):
        return np.sum(-y * np.log2(y_hat + int(y_hat == 0)*0.0001) - (1-y) * np.log2(1 - y_hat + int(y_hat == 1)*0.0001)) / y.shape[0]
    
    def get_error(self, y_hat, data):
        return self.logistic_cost(y_hat, data)
    
    def get_accuracy(self, data):
        accuracy = 0
        predicted = self.predict(data)
        for i in range(len(data)):
            accuracy += (predicted[i][-1] == data[i][-1]) / len(data)
        return accuracy

class classification_forest():
    def __init__(self, data, tree_amt, max_depth = 10000, amt_features = "Not given"):
        if amt_features == "Not given":
            amt_features = int(np.sqrt(len(data[0]) - 1) + 2)
        sample_size = len(data) // tree_amt + 1
        self.trees = []
        for i in range(tree_amt):
            curr_data = []
            for j in range(sample_size):
                curr_data.append(data[np.random.randint(0, len(data))])
            features = []
            available_features = [i for i in range(1, len(data[0]) - 1)]
            for i in range(amt_features):
                features.append(np.random.randint(0, len(available_features)))
            self.trees.append(classification_tree(curr_data, max_depth))
    
    def vote(self, data_point):
        pred = 0
        for i in range(len(self.trees)):
            pred += self.trees[i].parse_tree(data_point, self.trees[i].tree)[0][-1]
        return pred >= 0.5
    
    def pred(self, data):
        for i in range(len(data)):
            data[i][0] = i
        pred = [0 for i in range(len(data))]
        for i in range(len(self.trees)):
            tree_pred = self.trees[i].predict(data)
            for j in tree_pred:
                pred[j[0]] += j[-1] / len(self.trees)
        for i in range(len(pred)):
            pred[i] = pred[i] >= 0.5
        return pred
    
    def logistic_cost(self, y_hat, y):
        return np.sum(-y * np.log2(y_hat + int(y_hat == 0)*0.0001) - (1-y) * np.log2(1 - y_hat + int(y_hat == 1)*0.0001)) / y.shape[0]
    
    def get_accuracy(self, data):
        accuracy = 0
        predicted = self.pred(data)
        for i in range(len(predicted)):
            accuracy += (predicted[i] == data[i][-1]) / len(data)
        return accuracy
    
class regression_forest():
    def __init__(self, data, tree_amt, max_depth = 10000, amt_features = "Not given"):
        if amt_features == "Not given":
            amt_features = int(np.sqrt(len(data[0]) - 1) + 2)
        sample_size = len(data) // tree_amt + 1
        self.trees = []
        for i in range(tree_amt):
            curr_data = []
            for j in range(sample_size):
                curr_data.append(data[np.random.randint(0, len(data))])
            features = []
            available_features = [i for i in range(1, len(data[0]) - 1)]
            for i in range(amt_features):
                features.append(np.random.randint(0, len(available_features)))
            self.trees.append(regression_tree(curr_data, max_depth))
    
    def vote(self, data_point):
        pred = 0
        for i in range(len(self.trees)):
            pred += self.trees[i].parse_tree(data_point, self.trees[i].tree)[0][-1] / len(self.trees)
        return pred
    
    def pred(self, data):
        for i in range(len(data)):
            data[i][0] = i
        pred = [0 for i in range(len(data))]
        for i in range(len(self.trees)):
            tree_pred = self.trees[i].predict(data)
            for j in tree_pred:
                pred[j[0]] += j[-1] / len(self.trees)
        return pred
    
    def MSE(self, y_hat, data):
        if len(data) > 0:
            MSE = 0
            for i in range(len(data)):
                MSE += (data[i][-1] - y_hat[i][-1])**2
            MSE = MSE / len(data)
            return MSE
        else:
            return 0
    def get_MSE(self, data):
        pred = self.pred(data)
        return self.MSE(pred, data)


def get_split(data):
    train = int(len(data) * 0.7)
    return data[train:], data[:train]
    
class Gradient_boosted_regression(regression_forest):
    def pred(self, data):
        for i in range(len(data)):
            data[i][0] = i
        pred = [0 for i in range(len(data))]
        for i in range(len(self.trees)):
            tree_pred = self.trees[i].predict(data)
            for j in tree_pred:
                pred[j[0]] += j[-1]
        return pred
    
    def vote(self, data_point):
        pred = 0
        for i in range(len(self.trees)):
            pred += self.trees[i].parse_tree(data_point, self.trees[i].tree)[0][-1] * self.gamma[i]
        return pred
    
    class constant:
        def __init__(self, value):
            self.value = value
        def parse_tree(self, datapoint):
            return self.value
        def predict(self, data):
            for i in range(len(data)):
                data[i][-1] = self.value
            return data

    def __init__(self, data, tree_amt, max_depth):
        #First of all, lets determine the constant value ("gamma") we initialize our value with:
        #This is done by differentiating: [1/m sum_1^m(y_i - gamma)**2]' = 2/m * sum(y_i - gamma) * (-1) = 0, then setting it to 0
        #And then solve this by dividing by -1:
        # 2/m * sum(y_i - gamma) = 2/m * sum(y_i) - 2/m * m * gamma  = 0; dividing by 2/m
        # sum(y_i) - m * gamma = 0 <=> gamma = sum(y_i) / m, thus making gamma just the average:
        gamma = 0
        for i in range(len(data)):
            gamma += data[i][-1] / len(data)
        const = self.constant(gamma)
        self.trees = [const]
        self.gamma = [1]
        #Now, we solve for the pseudo residuals (as in we dont actually calculate the derivative, but rather pretend f(x_i) = z_i to be some constant (such as gamma) which derivative is 0 ):
        # r_im = - d_J(y, f(x)) / d_f(x) before adding each new tree. As we already established, that derivative is: 2/m * sum(y_i - predicted) * (-1) 
        # Since '-' and (-1) both cancel out, we get: r_im = 2/m * sum(y_i - predicted)
        # Then, we will solve for gamma the same way as before
        # Additionally, we will regularize using shrinkage by introducing some learning rate v : 0 < v <= 1
        for i in range(tree_amt):
            r_im = []
            #Getting the new dataset ready, where we replace the actual values with the residuals:
            for i in range(len(data)):
                r_im.append(data[i])
                r_im[-1] = (data[i][-1] - self.vote(data[i])) * 2
            self.trees.append(regression_tree(r_im, max_depth))
            #Now, we gotta just figure out the gamma for that one, which we do the same way as before, only that we have:
            # 2/m * sum(y_i - vote_before + vote_last_tree*gamma * vote_last_tree) = 2/m * sum(y_i) - 2/m * m * gamma  = 0
            # = 2/m * sum(y_i - vote_before) 2 