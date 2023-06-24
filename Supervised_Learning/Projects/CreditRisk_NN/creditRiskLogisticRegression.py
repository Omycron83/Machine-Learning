#trying ordinary logistic regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#first lets get the data and prepare it:
data = pd.read_csv('credit_risk_dataset.csv').dropna()
#initializing our research data np matrix with the bias column
research_data = [np.ones(len(data.index))]
#first the different features we got
#we convert all columns to strings with their respective names
for col in data.columns:
    globals()['%s' % col] = data[col].to_numpy()
#now we need to convert all categories (for example in person_home_ownership where there is rent, own etc.) to numbers corresponding to them in order to
#be able to work with them
    if globals()['%s' % col].dtype == 'object':
        categories = np.unique(globals()['%s' % col])
        number = -1
        for i in categories:
            number += 1
            globals()['%s' % col] = [int(number) if x == i else x for x in globals()['%s' % col]]
    globals()['%s' % col] = np.array(globals()['%s' % col]).astype('float64')
    research_data = np.vstack([research_data, globals()['%s' % col]])
#as we now have a bunch of column vectors and we want our data to be in rows, lets take the transpose of our matrix
research_data = research_data.T
#now we gotta split that into our features and our prediction values (i.e. if the person defaults on this loan or not)
#we figure out the column our relevant feature is in (this is loan_status here), and then splice that out while adding together the remaining splices around it column-wise
X = np.concatenate((research_data[:, :9], research_data[:, 10:]), 1)
Y = research_data[:, 9]
#now, lets normalize the data:
for i in range(1, len(X[0, :].T)):
    X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
#now it turns out that we have a skewed dataset (around 80% of the people dont default), which lead the algorithm to, in the first tries, just predict everything to NOT default
#as that obviously is a shitty way to actually predict if someone will default (yeah, he probably wont so give that guy with 0 income and a past default who uses this to repay other debt the loan)
#we will now just conveniently cut out a bunch of non-default examples and train with that, testing with the all-including dataset afterwards (we will first shuffle the data to avoid any problems with potential data)
p = np.random.permutation(len(Y))
X = X[p]
X_real = np.copy(X)
Y = Y[p]
Y_real = np.copy(Y)
#then see how many examples we will need to cut out in order to get to 50/50 default/no default
n = - (len(Y) - 2 * np.sum(Y))
#then, as long as we havent deleted enough rows yet we figure out if a row is non-default, and if it is add it to a list of the should-be deleted indeces (as doing it right there would have other problems i have no idea and dont want to get into fixing rn as they would require me to research a bunch of stuff)
delets = []
for i in range(len(Y)):
    if n < 0:
        if Y[i] == 0:
            n += 1
            delets.append(i)
    else:
        break
Y = np.delete(Y, delets, axis = 0)
X = np.delete(X, delets, axis = 0)
print(np.sum(Y) / len(Y))

def sigmoid(X):
    sig = 1 / (1 + np.exp(-X))
    sig = np.minimum(sig, 0.9999)
    sig = np.maximum(sig, 0.0001)
    return sig
alpha = 0.21

theta = np.random.rand(12, 1)
print(X @ theta)

for i in range(1000):
    J = ((-np.transpose(Y)@np.log(sigmoid(X@theta)))  -  (1-np.transpose(Y))@np.log(1 - sigmoid(X@theta)))/ len(Y)
    theta = [theta[j] - alpha * ((X @ theta - Y) * X[:, j]) / len(Y) for j in range(len(theta))]
    if i % 10 == 0:
        print("At iteration", i, "cost", J)

