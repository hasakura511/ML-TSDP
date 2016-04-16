import cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
import pandas.io.data
import re
from dateutil import parser

import datetime

import numpy as np
import matplotlib.pyplot as plt
try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

from os import listdir
from os.path import isfile, join
import re
import pandas as pd

def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    #dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
    dataframe[roll_n] = dataframe[returns].rolling(window=n, center=False).mean()

    
def applyRollMeanDelayedReturns(datasets, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    
    return datasets    

def mergeDataframes(datasets, index, cut):
    """
    merges datasets in the list 
    """
    #subset = []tion
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    
    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
    finance = finance[finance.index > cut]
    return finance

def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    
    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]