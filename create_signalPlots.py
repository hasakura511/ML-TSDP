# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:56:25 2016

@author: Hidemi
"""

import math
import string
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.dates as mdates
from pytz import timezone
from datetime import datetime as dt
from os import listdir
from os.path import isfile, join
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.color_palette("Set1", n_colors=8, desat=.5)

start_time = time.time()
size = (12,13)
systems = ['v1','v2','v3']

if len(sys.argv) > 1:
    bestParamsPath = './data/params/'
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    savePath = './data/signalPlots/'
    showPlot = False
else:
    signalPath = 'D:/ML-TSDP/data/signals/' 
    #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    savePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    showPlot = True
    
def fixnans(dataSet):
    fixed = np.zeros(dataSet.shape[0])
    for i,s in enumerate(dataSet.values):
        #print i,s,
        if np.isnan(s):
            fixed[i] = fixed[i-1]
        else:
            fixed[i] = s
        #print fixed[i]
    return fixed
    
def calcEquity_signals(SST, title, leverage=1.0, savePath=None, figsize=(8,7), showPlot=True):
    initialEquity = 1.0
    nrows = SST.gainAhead.shape[0]
    #signalCounts = SST.signals.shape[0]
    print '\nThere are %0.f signal counts' % nrows
    if 1 in SST.signals.value_counts():
        print SST.signals.value_counts()[1], 'beLong Signals',
    if -1 in SST.signals.value_counts():
        print SST.signals.value_counts()[-1], 'beShort Signals',
    if 0 in SST.signals.value_counts():
        print SST.signals.value_counts()[0], 'beFlat Signals',
        
    equityCurves = {}
    for trade in ['l','s','b']:       
        trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
        numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
        equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
        maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
        drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
        maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
        safef = pd.Series(data=leverage,index=range(0,len(SST.index)),name='safef')

        for i in range(0,len(SST.index)):
            if i == 0:
                equity[i] = initialEquity
                trades[i] = 0.0
                numBars[i] = 0.0
                maxEquity[i] = initialEquity
                drawdown[i] = 0.0
                maxDD[i] = 0.0

            else:
                if trade=='l':
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1 
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])

                        #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                elif trade=='s':
                    if (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    elif (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                        
        SSTcopy = SST.copy(deep=True)
        if trade =='l':
            #changeIndex = SSTcopy.signals[SST.signals==-1].index
            SSTcopy.loc[SST.signals==-1,'signals']=0
        elif trade =='s':
            #changeIndex = SSTcopy.signals[SST.signals==1].index
            SSTcopy.loc[SST.signals==1,'signals']=0
            
        equityCurves[trade] = pd.concat([SSTcopy.reset_index(), safef, trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)

    #  Compute cumulative equity for all days (buy and hold)   
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    safef = pd.Series(data=1.0,index=range(0,len(SST.index)),name='safef')
    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            numBars[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0
        else:
            trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
            numBars[i] = numBars[i-1] + 1 
            equity[i] = equity[i-1] + trades[i]
            maxEquity[i] = max(equity[i],maxEquity[i-1])
            drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
            maxDD[i] = max(drawdown[i],maxDD[i-1])          
    SSTcopy.loc[SST.signals==-1,'signals']=1
    SSTcopy.loc[SST.signals==0,'signals']=1

    equityCurves['buyHold'] = pd.concat([SSTcopy.reset_index(), safef, trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)

    if not SST.index.to_datetime()[0].time() and not SST.index.to_datetime()[1].time():
        barSize = '1 day'
    else:
        barSize = '1 min'
        
    #plt.close('all')
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=figsize)
    #plt.subplot(2,1,1)
    ind = np.arange(SST.shape[0])
    ax1.plot(ind, equityCurves['l'].equity,label="Long 1 Signals",color='b')
    ax1.plot(ind, equityCurves['s'].equity,label="Short -1 Signals",color='r')
    ax1.plot(ind, equityCurves['b'].equity,label="Long & Short",color='g')
    ax1.plot(ind, equityCurves['buyHold'].equity,label="BuyHold",ls='--',color='c')
    #fig, ax = plt.subplots(2)
    #plt.subplot(2,1,2)
    ax2.plot(ind, -equityCurves['l'].drawdown,label="Long 1 Signals",color='b')
    ax2.plot(ind, -equityCurves['s'].drawdown,label="Short -1 Signals",color='r')
    ax2.plot(ind, -equityCurves['b'].drawdown,label="Long & Short",color='g')
    ax2.plot(ind, -equityCurves['buyHold'].drawdown,label="BuyHold",ls='--',color='c')
    
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)

    if barSize != '1 day' :
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d %H:%M")
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    else:
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the
    # toolbar

    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.set_title(title)
    ax1.set_ylabel("TWR")
    ax1.legend(loc="best")
    ax2.set_ylabel("Drawdown")   
    if savePath != None:
        plt.savefig(savePath+title+'.png', bbox_inches='tight')
        
    if showPlot:
        plt.show()

    
    shortTrades, longTrades = numberZeros(SST.signals)
    allTrades = shortTrades+ longTrades
    print '\nValidation Period from', SST.index[0],'to',SST.index[-1]
    print 'TWR for Buy & Hold is %0.3f, %i Bars, maxDD %0.3f' %\
                (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
    print 'TWR for %i beLong trades is %0.3f, maxDD %0.3f' %\
                (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
    print 'TWR for %i beShort trades is %0.3f, maxDD %0.3f' %\
                (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
    print 'TWR for %i beLong and beShort trades is %0.3f, maxDD %0.3f' %\
                (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
    print 'SAFEf:', equityCurves['b'].safef.mean()

    SST_equity = equityCurves['b']
    if 'dates' in SST_equity:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['dates'])).drop(['dates'], axis=1)
    else:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['index'])).drop(['index'], axis=1)
        

    
signalFiles = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
dataFiles = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
pairs = [x[6:12] for x in dataFiles]

for pair in pairs:
    dataFilename = [pairs for pairs in dataFiles if pair in pairs][0]
    dataFile = pd.read_csv(dataPath + dataFilename, index_col='Date').drop_duplicates()
    print 'Loaded data from', dataFilename
    
    validSignalFiles = {}
    for system in systems: 
        validSignalFiles[system]=[x for x in signalFiles if system in x]
    #for f in [sfile for sfilelist in validSignalFiles for sfile in sfilelist]:
    for system in validSignalFiles:
        if system != 'v1':
            for f in [sfile for sfile in validSignalFiles[system] if pair in sfile]:
                signalFile = pd.read_csv(signalPath+f, index_col='dates').drop_duplicates()
                print 'Loaded signals from', f
                #if there is no prior index in the row, then it's a legacy file
                if 'prior_index' in signalFile:
                    #prior index is the key for valid signal rows
                    sst = signalFile[signalFile.prior_index != 0].sort_index()
                    
                    #remove rows with duplicate indices
                    reindexed_sst = pd.DataFrame()
                    for i,x in enumerate(sst.index):
                        if i ==0:
                            reindexed_sst = reindexed_sst.append(sst.iloc[i])
                        elif x != reindexed_sst.index[-1]:
                            reindexed_sst = reindexed_sst.append(sst.iloc[i])
                        else:
                            #case where last index was a dupe
                            pass
                    dataFile.index = dataFile.index.to_datetime()
                    reindexed_sst.index = reindexed_sst.index.to_datetime()
                    intersect = reindexed_sst.index.to_datetime()\
                                            .intersection(dataFile.index.to_datetime())
                                            
                    dataSet = pd.concat([reindexed_sst,\
                                    dataFile.ix[intersect[0]:]],join = 'outer',axis=1)
                    dataSet['gainAhead'] = gainAhead(dataSet.Close)
                    dataSet['signals'] = fixnans(dataSet.signals)
                    dataSet['safef'] = fixnans(dataSet.safef)
                    dataSet = dataSet[['signals','gainAhead','safef']].dropna()
                    
                    equityCurve = calcEquity_signals(dataSet, f[:-4],\
                                        leverage = dataSet.safef.values, savePath=savePath,\
                                        figsize=size, showPlot=showPlot)
                    equityCurve.to_csv(savePath+f[:-4]+'_curve.csv')
        else:
            for f in [sfile for sfile in validSignalFiles[system] if pair in sfile]:
                signalFile = pd.read_csv(signalPath+f, index_col='dates').drop_duplicates()
                print 'Loaded signals from', f
                #if there is no prior index in the row, then it's a legacy file
                if 'prior_index' in signalFile:
                    #prior index is the key for valid signal rows
                    sst = signalFile.sort_index()
                    
                    #remove rows with duplicate indices
                    reindexed_sst = pd.DataFrame()
                    for i,x in enumerate(sst.index):
                        if i ==0:
                            reindexed_sst = reindexed_sst.append(sst.iloc[i])
                        elif x != reindexed_sst.index[-1]:
                            reindexed_sst = reindexed_sst.append(sst.iloc[i])
                        else:
                            #case where last index was a dupe
                            pass
                    dataFile.index = dataFile.index.to_datetime()
                    reindexed_sst.index = reindexed_sst.index.to_datetime()
                    intersect = reindexed_sst.index.to_datetime()\
                                            .intersection(dataFile.index.to_datetime())
                                            
                    dataSet = pd.concat([reindexed_sst,\
                                    dataFile.ix[intersect[0]:]],join = 'outer',axis=1)
                    dataSet['gainAhead'] = gainAhead(dataSet.Close)
                    dataSet['signals'] = fixnans(dataSet.signals)
                    dataSet['safef'] = fixnans(dataSet.safef)
                    dataSet = dataSet[['signals','gainAhead','safef']].dropna()
                    
                    equityCurve = calcEquity_signals(dataSet, f[:-4],\
                                        leverage = dataSet.safef.values, savePath=savePath,\
                                        figsize=size, showPlot=showPlot)
                    equityCurve.to_csv(savePath+f[:-4]+'_curve.csv')
                
                
                            
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
            
            

