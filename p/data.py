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
from dateutil.parser import parse
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
import os
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import seitoolz.bars as bars

def getStock(path_datasets, symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  pd.io.data.get_data_yahoo(symbol, start, end)
    df.to_csv(path_datasets[0]+'idx_' + symbol + '.csv')
    
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
    
    bars.bidask_to_csv('ES', out.index[-1], out['Close'][-1], out['Close'][-1])
    
    #out.columns.values[-1] = 'AdjClose'
    #out.columns = out.columns + '_Out'
    #out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]


def loadDatasets(path_datasets, fout, parameters):
    interval=parameters[1]
    dataSet=list()
    count=0
    
    for dataPath in path_datasets:
        symbol_dict=dict()
        files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
        for file in files:
            
            fsym=fout
            ireg=re.search(r'\W',fout)
            if ireg:
                fsym=fout.split(ireg.group(0))[1]
            for inter in interval:
                if re.search(r''+inter, file) and not re.search(fsym,file):
                    sym=file.rsplit('.',1)[0]
                    symbol_dict[sym]=sym
                    print 'loadDatasets: ', sym
        symbols, names = np.array(list(symbol_dict.items())).T
        print 'loadDatasets: Main Symbol', fout
        
        data = list()
        maxdate=datetime.datetime(1970,1,1)
        for symbol in symbols:
            dataFrame=get_quote(dataPath, symbol, symbol, True, parameters)
            dataFrame.index=pd.to_datetime(dataFrame.index)
            dataFrame=dataFrame.sort_index()   
            
            if dataFrame.shape[0]>1000:
                if dataFrame.index[0] > maxdate:
                    maxdate=dataFrame.index[0];
                    print symbol, ' Start: ', dataFrame.index[0], 'Start Period:',maxdate
                data.append(dataFrame)
            
        if count == 0:
            dataFrame =  get_quote(dataPath, fout, 'Out', True, parameters)
            if dataFrame.index[0] > maxdate:
                maxdate=dataFrame.index[0];
            dataSet.append(dataFrame)
        #for df in data:
        #    df=df[df.index >= maxdate]
        #    if df.shape[0]>1000:
        #        dataSet.append(df)
        dataSet.extend(data)
        dataSet=[ df[df.index >= maxdate] for df in dataSet ]
        count = count + 1
    print 'Done Loading Data'
    return dataSet

def reset_cache():
    global qcache
    qcache=dict()
qcache=dict()
def get_quote(dataPath, sym, colname, addParam, parameters):
    global qcache
    filename=dataPath + sym + '.csv'
    if not qcache.has_key(sym):
        if os.path.isfile(filename):
            dataSet=pd.read_csv(filename,index_col='Date')
            if not dataSet.shape[0]>1000:
                return dataSet
        else:
            filename=dataPath + sym + '.txt'
            dataSet=pd.read_csv(filename)
            dataSet=dataSet.rename(columns=lambda x: x.strip()[0:1].upper() + x.strip()[1:].lower())
            #for x in dataSet.columns.copy():
                #if x not in ['Date','Open','High','Low','Close','Volume','Vol']:
                #    dataSet=dataSet.drop(x, axis=1)
            dataSet['Date']=[parse(str(x)) for x in dataSet['Date']]
            print dataSet.columns
            dataSet=dataSet.set_index('Date')
    
        dataSet.index=pd.to_datetime(dataSet.index)
        dataSet=dataSet.sort_index()    
        qcache[sym]=dataSet.copy()
    else:
        dataSet=qcache[sym].copy()
    if parameters[2] > 0:
        dataSet=dataSet.iloc[:-parameters[2]].copy().sort_index();
        
    if addParam:
        
        ###### Future #######
        data=dataSet.iloc[-1].copy()
        data=dataSet.reset_index().iloc[-1].copy()
        #data['Open']=(data['High'] + data['Low'])/2
        data['Close']=data['Open']
        #data['Open']=data['Close']
        #data['High']=data['Open']
        #data['Low']=data['Open']
        #data['Volume']=0
        if parameters[1] == '30m_': 
            data['Date']=data['Date']+datetime.timedelta(minutes=30)
        elif parameters[1] == '1h_': 
            data['Date']=data['Date']+datetime.timedelta(hours=1)
        elif parameters[1] == '10m_': 
            data['Date']=data['Date']+datetime.timedelta(minutes=10)
        elif parameters[1] == '1 min_': 
            data['Date']=data['Date']+datetime.timedelta(minutes=1)
        elif parameters[1] == 'BTCUSD_':
            data['Date']=data['Date']+datetime.timedelta(minutes=1)
        else:
            data['Date']=data['Date']+datetime.timedelta(days=1)
        ###### DataSet ######
        dataSet=dataSet.reset_index().append(data).set_index('Date')
        #dataSet.columns.values[-1] = 'AdjClose'
        dataSet['AdjClose']=dataSet['Close']
        #dataSet['Open']=dataSet['Open']
        dataSet.columns = dataSet.columns + '_' + colname
        dataSet['Return_%s' %colname] = dataSet['AdjClose_%s' %colname].pct_change()
    #dataSet=dataSet.ix[dataSet.index[-1] - datetime.timedelta(days=10):]
    print 'Loaded '+ sym + ' ' + str(dataSet.shape[0]) + ' Rows'
   
    return dataSet
