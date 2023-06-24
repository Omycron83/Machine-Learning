#monte carlo stock market simulation:
#idea: we will figure out the distribution of the returns of SPY over a couple of decades and then simulate multiple outcomes
#the flaw though: stock returns may not be independent, as there may be trends (e.g. in a bear market, stock returns may be skewed to the left (negative), while in a bull the opposite)
#first, lets load in the data:
import datetime as dt
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
#variables to play around with
inputMoney = 10
forecastYears = 100
samples = 500
years = 50
#loading in the data
data = pdr.get_data_yahoo("SPY", (dt.datetime.now() - dt.timedelta(days=years * 365)), dt.datetime.now(), interval = "d")
SPY = data["Adj Close"].to_list()
#getting the returns
SPYReturns = [SPY[x] / SPY[x - 1] - 1 for x in range(1, len(SPY))]
print(np.mean(SPYReturns), np.median(SPYReturns))
print("Mean expected value:", ((np.mean(SPYReturns) + 1)**365)**forecastYears * inputMoney)
#sampling n samples with the size of the amount of years we have with replacement, utilizing the bootstrap
X = np.random.choice(SPYReturns, forecastYears * 365)
for i in range(samples):
    X = np.vstack((X, np.random.choice(SPYReturns, forecastYears * 365, replace = True)))
X = np.c_[np.ones((np.shape(X)[0], 1)) * inputMoney, X]
#now we simulate how the amount of invested money would have behaved given the sampled returns
for i in range(1, forecastYears * 365 + 1):
    X[:, i] = X[:, i - 1] * (X[:, i] + 1)

#plotting the results:
for i in range(len(X[:, 1])):
    #random color so that we can distinguish the lines ^^
    col = tuple(np.random.randint(100, size=3) / 100)
    plt.plot(np.linspace(0, forecastYears*365 + 1, forecastYears*365 + 1), X[i, :], color = col)

plt.show()
#showing the histogramm of the returns we would could have gotten
plt.hist(X[:, -1] , int(samples / 10))
plt.show()
print("Median:", np.median(X[:, -1]),"Mean:", np.average(X[:, -1]),"Std:", np.std(X[:, -1]),"Min:", min(X[:, -1]),"Max:", max(X[:, -1]), "SE:", np.std(X[:, -1])/ np.sqrt(samples))
