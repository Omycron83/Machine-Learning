import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import numpy as np
pd.set_option('display.max_columns', 10)
data = (pdr.get_data_yahoo("SPY", (dt.datetime.now() - dt.timedelta(days=7300)), dt.datetime.now()))#.to_numpy()
#lets first calculate...elasticity of volume in SPY (aka, how does a change in volume respond to a change in price?)
price = data["Adj Close"].tolist()
volume = data["Volume"].tolist()
avgPriceChange = 0
avgVolumeChange = 0
for i in range(len(price)-1): 
    deltaPrice = price[i] - price[i + 1]
    midpointPrice = deltaPrice / ((price[i] + price[i+1])/2)
    avgPriceChange += midpointPrice / len(price)
    deltaVolume = volume[i] - volume[i + 1]
    midpointVolume = deltaVolume / ((volume[i] + volume[i+1])/2)
    avgVolumeChange += midpointVolume / len(price)
elasticityOfVolume = avgPriceChange / avgVolumeChange
print(elasticityOfVolume)
#interesting, about -1...so basically the change in volume is antiproportional to the one of the price, aka the higher the price the lower the volume, which means laws of supply and demand apply?
print(avgPriceChange * (-1))
#so on kind of a fake average, the price of SPY has risen at around 0.0003% on every day
