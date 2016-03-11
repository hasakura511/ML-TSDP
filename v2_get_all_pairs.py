# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016

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
from pytz import timezone
from datetime import datetime as dt


#other
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV

#suztoolz
from suztoolz.display import sss_display_cmatrix, is_display_cmatrix2,\
                         oos_display_cmatrix2, init_report, update_report,\
                         showPDF, showCDF, getToxCDF, plot_learning_curve,\
                         directional_scoring, compareEquity, describeDistribution
from suztoolz.loops import sss_iterate_train, adjustDataProportion, CAR25_df,\
                            maxCAR25, wf_regress_validate2, sss_regress_train, calcDPS2,\
                            calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
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
cycles = 2
exchange='IDEALPRO'
symbol='AUD'
currency='USD'
secType='CASH'
endDateTime=dt.now(timezone('US/Eastern'))
durationStr='1 M'
barSizeSetting='30 mins'
whatToShow='MIDPOINT'
ticker = symbol + currency

currencyPairs = ['AUDUSD','EURUSD','GBPUSD','USDCAD',\
                'USDCHF','USDJPY','EURJPY']

############################################################
for i in range(0,cycles):
    if i ==0:
        getHistLoop = [endDateTime]
    else:
        getHistLoop.insert(0,(getHistLoop[0]-datetime.timedelta(365/12)))

getHistLoop = [x.strftime("%Y%m%d %H:%M:%S %Z") for x in getHistLoop]

currencyPairsDict = {}
for pair in currencyPairs:
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
    for date in getHistLoop:
        brokerData['client_id']=random.randint(100,1000)
        data = pd.concat([data,getDataFromIB(brokerData, date)],axis=0)
        time.sleep(3)
    currencyPairsDict[pair] = data
    data.to_csv(dataPath+'raw_'+pair+'.csv')
print 'Successfully Retrieved Data and saved data.'
###########################################################
