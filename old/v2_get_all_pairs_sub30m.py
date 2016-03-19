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

#suztoolz

from suztoolz.data import getDataFromIB

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
#endDateTime=dt.now(timezone('US/Eastern'))
endDateTime=dt.now(get_localzone())
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'
ticker = symbol + currency

currencyPairs = ['AUDUSD','EURUSD','GBPUSD','USDCAD',\
                'USDCHF','USDJPY','EURJPY']
'''
reqDuration = {
                '1 M':30,
                '2 W':14,
                '1 W':7,
                '1 D':1
                }
############################################################
for i in range(0,cycles):
    if i ==0:
        getHistLoop = [endDateTime]
    else:
        getHistLoop.insert(0,(getHistLoop[0]-\
                                datetime.timedelta(reqDuration[durationStr])))
'''
#getHistLoop = []
#getHistLoop = [endDateTime.strftime("%Y%m%d %H:%M:%S %Z") for x in getHistLoop]

currencyPairsDict = {}
for pair in currencyPairs:
    #turn this off to get progressively older dates for testing.
    
    eastern=timezone('US/Eastern')
    date=endDateTime.astimezone(eastern)
    date=date.strftime("%Y%m%d %H:%M:%S EST")
    symbol = pair[0:3]
    currency = pair[3:6]
    brokerData = {}
    brokerData =  {'port':7496, 'client_id':101,\
                         'tickerId':1, 'exchange':exchange,'symbol':symbol,\
                         'secType':secType,'currency':currency,\
                         'endDateTime':endDateTime, 'durationStr':durationStr,\
                         'barSizeSetting':barSizeSetting,\
                         'whatToShow':whatToShow, 'useRTH':1, 'formatDate':1
                          }
                          
    data = pd.DataFrame()
    #for date in getHistLoop:
    while data.shape[0] < minDataPoints:      
        brokerData['client_id']=random.randint(100,1000)
        requestedData = getDataFromIB(brokerData, date)
        #update date
        date = endDateTime.tzinfo.localize(requestedData.index.to_datetime()[0]\
                .to_pydatetime())
        eastern=timezone('US/Eastern')
        date=date.astimezone(eastern)
        date=date.strftime("%Y%m%d %H:%M:%S EST")
        if len(data)==0:
            data = pd.concat([requestedData,data],axis=0)
        else:
            if sum(pd.concat([requestedData,data],axis=0).duplicated())<100:
                data = pd.concat([requestedData,data],axis=0)
        #set date as last index for next request

        time.sleep(3)
        
    currencyPairsDict[pair] = data
    data.to_csv(dataPath+barSizeSetting+'_'+pair+'.csv')
    print 'Successfully Retrieved Data and saved',data.shape[0],'rows of data.\n\n'
    
###########################################################
