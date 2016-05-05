# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import math
import talib as ta
import pandas as pd
from suztoolz.transform import zigzag as zg
import arch
from os import listdir
from os.path import isfile, join

from datetime import datetime
import matplotlib.pyplot as plt
#from pandas.io.dataSet import DataReader
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import time
from suztoolz.transform import perturb_data
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio

from suztoolz.loops import CAR25, CAR25_prospector, maxCAR25
from suztoolz.display import init_report, update_report_prospector,\
                            display_CAR25, compareEquity_vf, getToxCDF, adf_test,\
                            describeDistribution
from sklearn.grid_search import ParameterGrid
import re
import copy
import string
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import datetime
from datetime import datetime as dt
from pandas.core import datetools
import time
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter, getCycleTime, saveParams

from suztoolz.loops import calcDPS2, calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.display import displayRankedCharts,offlineMode
from sklearn.preprocessing import scale, robust_scale, minmax_scale
import logging
import os
from pytz import timezone
from dateutil.parser import parse

def loadFutures(auxFutures, dataPath, barSizeSetting, maxlb, ticker,\
                                signalPath, version, version_, maxReadLines,\
                                **kwargs):
    perturbData =  kwargs.get('perturbData',False)
    perturbDataPct=kwargs.get('perturbDataPct',0.0002)
    verbose=kwargs.get('verbose',False)
    addAux=kwargs.get('addAux',False)
    futuresDict = {}
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

    if addAux:    
        for contract in auxFutures:    
            #if 'F_'+contract+'.txt' in files and (ticker[0:3] in contract or ticker[3:6] in contract):
            data = pd.read_csv(dataPath+'F_'+contract+'.txt', index_col=0)
            data = data.drop([' P',' R', ' RINFO'],axis=1)
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','Volume','OI']
            if data.shape[0] < maxlb:
                if contract == ticker:
                    message =  'Not enough data to create indicators: #rows\
                        is less than max lookback of '+str(maxlb)
                    offlineMode(ticker, message, signalPath, version, version_)
                print( 'Skipping aux contract '+contract+'. Not enough data to create indicators: '+\
                                    str(data.shape[0])+' rows is less than max lookback of '+str(maxlb))
            elif data.shape[0] >= maxReadLines:
                futuresDict[contract] = data[-maxReadLines:]
            else:
                futuresDict[contract] = data
                    
        nrows = futuresDict[ticker].shape[0]
        
        #print 'removing dupes..'
        futuresDict2 = copy.deepcopy(futuresDict)
        for contract in futuresDict2:
            data = futuresDict2[contract].copy(deep=True)
            #perturb dataSet
            if perturbData:
                if verbose:
                    print 'perturbing OHLC and dropping dupes',
                data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
                data['High']= perturb_data(data['High'].values,perturbDataPct)
                data['Low']= perturb_data(data['Low'].values,perturbDataPct)
                data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
                
            if verbose:
                print 'Checking dupes:',contract, futuresDict2[contract].shape,'to',
            if len(futuresDict2[contract].ix[futuresDict2[contract].index.duplicated()])>0:
                print futuresDict2[contract].ix[futuresDict2[contract].index.duplicated()],
                futuresDict2[contract] = data.drop_duplicates()
            #print futuresDict2[contract].tail()
            if verbose:
                print contract, futuresDict2[contract].shape
            
        dataSet=futuresDict2[ticker].copy(deep=True)
        lastIndex = dataSet.index[-1]       
        
        if verbose:            
            for contract in futuresDict2:
                if dataSet.shape[0] != futuresDict2[contract].shape[0]:
                    print 'Warning:',ticker, contract, 'row mismatch. Some Data may be lost.'
                    
        #align the index, at the end of the loop, dataSet should have an index that all contracts contain
        for contract in futuresDict2:
            missingData=futuresDict2[ticker].index.sym_diff(futuresDict2[contract].index)
            print 'Missing data:',ticker,contract, missingData
            
            #forward fill missing data
            if len(missingData)>0:
                print 'Forward filling missing data..'
                #print futuresDict2[contract].ix[missingData]
                futuresDict2[contract]=futuresDict2[contract].ix[futuresDict2[ticker].index].fillna(method='ffill')
                futuresDict2[ticker]=futuresDict2[ticker].ix[futuresDict2[contract].index].fillna(method='ffill')
                #first index not forwardfillable
                futuresDict2[contract]=futuresDict2[contract].dropna()
                #print futuresDict2[contract].ix[missingData]         
                futuresDict2[ticker]=futuresDict2[ticker].dropna()
                
            if contract != ticker:
                intersect =np.intersect1d(futuresDict2[contract].index, dataSet.index)
                dataSet = dataSet.ix[intersect]
        #align the other contracts.
        for contract in futuresDict2:
            #print 'Reindex:',contract, futuresDict2[contract].shape,
            futuresDict2[contract] = futuresDict2[contract].ix[dataSet.index]
            #print 'to', futuresDict2[contract].shape
                
        dataSet=futuresDict2[ticker].copy(deep=True)
        print nrows-dataSet.shape[0], 'rows lost for', ticker
        print futuresDict[ticker].index.sym_diff(dataSet.index)
        
        futuresDict3 ={}
        for contract in futuresDict2:
            if contract !=ticker:
                closes = pd.concat([dataSet.Close, futuresDict2[contract].Close],\
                                        axis=1, join='inner')
                highs = pd.concat([dataSet.High, futuresDict2[contract].High],\
                                axis=1, join='inner')
                lows = pd.concat([dataSet.Low, futuresDict2[contract].Low],\
                                        axis=1, join='inner')
                #check if there is enough data to create indicators
                if closes.shape[0] < maxlb:
                    message = 'Not enough data to create indicators: intersect of '\
                        +ticker+' and '+contract +' of '+str(closes.shape[0])+\
                        ' is less than max lookback of '+str(maxlb)
                    offlineMode(ticker, message, signalPath, version, version_)
                else:
                    futuresDict3[contract] = {'closes':closes,'highs':highs,'lows':lows}
                    
        return dataSet, futuresDict3
    else:
        futuresDict = {}
        files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
        if 'F_'+ticker+'.txt' in files:
            data = pd.read_csv(dataPath+'F_'+ticker+'.txt', index_col=0)
            data = data.drop([' P',' R', ' RINFO'],axis=1)
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','Volume','OI']
            if data.shape[0] < maxlb:
                message =  'Not enough data to create indicators: #rows\
                    is less than max lookback of '+str(maxlb)
                offlineMode(ticker, message, signalPath, version, version_)
            elif data.shape[0] >= maxReadLines:
                futuresDict[ticker] = data[-maxReadLines:]
            else:
                futuresDict[ticker] = data

        nrows = futuresDict[ticker].shape[0]

        #print 'removing dupes..'
        futuresDict2 = copy.deepcopy(futuresDict)
        data = futuresDict2[ticker].copy(deep=True)
        #perturb dataSet
        if perturbData:
            if verbose:
                print 'perturbing OHLC and dropping dupes',
            data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
            data['High']= perturb_data(data['High'].values,perturbDataPct)
            data['Low']= perturb_data(data['Low'].values,perturbDataPct)
            data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
            
        if verbose:
            print ticker, futuresDict2[ticker].shape,'to',
        if len(futuresDict2[ticker].ix[futuresDict2[ticker].index.duplicated()])>0:
            print futuresDict2[ticker].ix[futuresDict2[ticker].index.duplicated()],
            futuresDict2[ticker] = data.drop_duplicates()
        #print futuresDict2[ticker].tail()
        if verbose:
            print ticker, futuresDict2[ticker].shape
                
        dataSet=futuresDict2[ticker].copy(deep=True)
        return dataSet, futuresDict2