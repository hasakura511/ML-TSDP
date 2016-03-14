
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history
from c2api.place_order import place_order as place_c2order
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone

#other
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV

start_time = time.time()


debug=False

if len(sys.argv)==1:
    debug=True

if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = './debug/scored_metrics_'
    equityStatsSavePath = './debug/'
    signalPath = './debug/'
    dataPath = './data/from_IB/'
else:
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'

    
#data Parameters
#cycles = 2
minDataPoints = 2000
exchange='IDEALPRO'
symbol='AUD'
currency='USD'
secType='CASH'
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'
ticker = symbol + currency

currencyPairs = ['NZDJPY','CADJPY','CHFJPY', \
                 'EURGBP','GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']


def get_ibfeed(sym, cur):
	get_feed(sym, cur,'IDEALPRO','CASH')

currencyPairsDict = {}
tickerId=1
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
for pair in currencyPairs:
    filename=dataPath+barSizeSetting+'_'+pair+'.csv'
    data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
   
    eastern=timezone('US/Eastern')
    endDateTime=dt.now(get_localzone())
    date=endDateTime.astimezone(eastern)
    date=date.strftime("%Y%m%d %H:%M:%S EST")
    
    symbol = pair[0:3]
    currency = pair[3:6]
    tickerId=tickerId+1
        
    currencyPairsDict[pair] = data
    
    get_realtimebar(symbol, currency,   exchange, secType, whatToShow, data, filename, tickerId)

    data=get_history(date, symbol, currency, exchange, secType, whatToShow, data, filename, tickerId, minDataPoints, durationStr, barSizeSetting)
   
finished=False
while not finished:
    #if iserror:
    #    finished=True
    #if (time.time() - start_time) > seconds: ## get ~4 samples over 15 seconds
    #    finished=True
    time.sleep(30)
    

