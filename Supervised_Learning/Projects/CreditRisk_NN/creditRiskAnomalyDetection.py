#credit risk anomaly detection
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
#for our anomaly detection, we want to get a vector of the mean and a vector of the Std's of each column:
probs = np.zeros((np.shape(X)[0], np.shape(X)[1] - 1))
for i in range(len(probs[0])):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)

def normalDistribution(X, var, mean):
    return 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-(X - mean)**2 / (2 * var))

for i in range(1, len(X[0])):
    probs[:, i - 1] = normalDistribution(X[:, i], var[i], mean[i])
predValues = np.prod(probs, axis = 1)

epsilons = []
f1 = []
for i in range(1):
    epsilon = 4e-17 #1 / 10**i
    tp = np.sum((predValues < epsilon) & (Y == 1))
    fp = np.sum((predValues < epsilon) & (Y == 0))
    fn = np.sum((predValues > epsilon) & (Y == 1))
    accuracy = tp / (tp + 1/2 * (fp + fn))
    epsilons.append(epsilon)
    f1.append(accuracy)
    print("This amounts to an accuracy at epsilon =", epsilon, "of:", accuracy)
    
#print(epsilons)
#print(f1)
#lowest = f1.index(max(f1))
#lowestEpsiol = epsilons[lowest]
#print(lowestEpsiol)
#print(lowest)
#print(f1[lowest])
#accuracy = np.sum((predValues < epsilon) == Y) / len(Y)
