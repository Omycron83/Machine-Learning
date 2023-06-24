#trying to calculate the beta of a stock (FB) (or any other for that matter)
import datetime as dt
import numpy as np
from pandas_datareader import data as pdr
years = 5
data = pdr.get_data_yahoo("SPY", (dt.datetime.now() - dt.timedelta(days=years * 365)), dt.datetime.now(), interval = "m")
SPY = data["Adj Close"].to_list()
data = pdr.get_data_yahoo("MSFT", (dt.datetime.now() - dt.timedelta(days=years * 365)), dt.datetime.now(), interval = "m")
FB = data["Adj Close"].to_list()
SPY = [SPY[i + 1] / SPY[i] - 1 for i in range(len(SPY) - 1)]
FB = [FB[i + 1] / FB[i] - 1 for i in range(len(FB) - 1)]
print(len(SPY), len(FB))
mean_SPY = sum(SPY) / len(SPY)
mean_FB = sum(FB) / len(FB)
_covariance = sum((x - mean_SPY)*(y - mean_FB) for x, y in zip(SPY, FB)) / (len(FB) - 1)
_variance = sum((x - mean_SPY)**2 for x in SPY) / (len(SPY) - 1)

Beta = _covariance/_variance
print(Beta)

Beta2 = 0
alpha = 10
iterations = 1000
theta = 0
for i in range(iterations):
    theta -= alpha/len(FB) * np.sum((np.array(SPY) * theta - np.array(FB)) * SPY)
    if i % 100 == 0:
        print("At i =", i, "theta =", theta ,"error =", np.sum((np.array(SPY) * theta - np.array(FB))**2)/len(SPY))

Beta2 = theta
print(Beta2)
