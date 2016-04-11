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
from sklearn.qda import QDA
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

def getStock(path_datasets, symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  pd.io.data.get_data_yahoo(symbol, start, end)
    df.to_csv(path_datasets+'idx_' + symbol + '.csv')
    
    #df.columns.values[-1] = 'AdjClose'
    #df.columns = df.columns + '_' + symbol
    #df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    
    return df

def getStockFromQuandl(symbol, name, start, end):
    """
    Downloads Stock from Quandl.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    import Quandl
    df =  Quandl.get(symbol, trim_start = start, trim_end = end, authtoken="your token")

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + name
    df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
    
    return df

def getStockDataFromWeb(fout, name, path_datasets, start_string, end_string):
    """
    Collects predictors data from Yahoo Finance and Quandl.
    Returns a list of dataframes.
    """
    start = parser.parse(start_string)
    end = parser.parse(end_string)
    
    nasdaq = getStock(path_datasets, '^IXIC', start, end)
    frankfurt = getStock(path_datasets, '^GDAXI', start, end)
    london = getStock(path_datasets, '^FTSE', start, end)
    paris = getStock(path_datasets, '^FCHI', start, end)
    hkong = getStock(path_datasets, '^HSI', start, end)
    nikkei = getStock(path_datasets, '^N225', start, end)
    australia = getStock(path_datasets, '^AXJO', start, end)
    djia = getStock(path_datasets, '^DJI', start, end) 
    out =  getStock(path_datasets, fout, start, end)
    #out.columns.values[-1] = 'AdjClose'
    #out.columns = out.columns + '_Out'
    #out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    
    
    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]


def loadDatasets(path_datasets, fout, interval):
    dataPath=path_datasets
    symbol_dict=dict()
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    for file in files:
        
        fsym=fout
        ireg=re.search(r'\W',fout)
        if ireg:
            fsym=fout.split(ireg.group(0))[1]
        #print 'loadDataSets: ' + fsym
        if re.search(r''+interval, file) and not re.search(fsym,file):
            sym=file.split('.')[0]
            symbol_dict[sym]=sym
            print sym
    symbols, names = np.array(list(symbol_dict.items())).T
    
    out =  get_quote(dataPath,fout, False)
    out['AdjClose']=out['Close']
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    out.to_csv('./p/data/out.csv')
    data = [get_quote(dataPath, symbol, True)
          for symbol in symbols]
    dataSet=list()
    dataSet.append(out)
    dataSet.extend(data)
    return dataSet

def get_quote(dataPath,sym, addParam=True):
    dataSet=pd.read_csv(dataPath + sym + '.csv',index_col='Date')
    dataSet.index=pd.to_datetime(dataSet.index)
    dataSet=dataSet.sort_index()    
    if addParam:
        #dataSet.columns.values[-1] = 'AdjClose'
        dataSet['AdjClose']=dataSet['Close']
        dataSet.columns = dataSet.columns + '_' + sym
        dataSet['Return_%s' %sym] = dataSet['AdjClose_%s' %sym].pct_change()
    #dataSet=dataSet.ix[dataSet.index[-1] - datetime.timedelta(days=10):]
    print 'Loaded '+ sym + ' ' + str(dataSet.shape[0]) + ' Rows'
    
    return dataSet
