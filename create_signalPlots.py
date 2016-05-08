# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:56:25 2016

@author: Hidemi
"""
import os
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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
size = (8,7)
versions = ['v4.3']
#versions = ['v1.3','v2.4']
barSize='1D'
#regime switching params
lookback = 90
with open('./data/futures.txt') as f:
    futures = f.read().splitlines()

                
if len(sys.argv) > 1:
    bestParamsPath = './data/params/'
    signalPath = './data/signals/'
    dataPath = './data/tickerData/'
    equityCurveSavePath = './data/signalPlots/'
    pngPath = './data/results/'
    showPlot = False
    verbose = False
else:
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    dataPath = 'D:/data/tickerData/'
    #bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    equityCurveSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 

    pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    showPlot = True
    verbose = True
    
def IBcommission(SST,i,tradeAmount,safef, asset):
    commission = 2.0
    if asset == 'FX':
        commission =  max(2.0, tradeAmount*2e-5)
        
    if i==0:
        #first trade
        return round(commission,2)
    elif SST.signals[i]*safef[i] !=SST.signals[i-1]*safef[i-1]:
        #signal change 0/-1 to 1
        return round(commission,2)
    else:
        #no signal change
        return 0.0

            
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
    
def calcEquity_signals(SST, title, **kwargs):
    #leverage=1.0, equityCurveSavePath=None, pngPath=None, figsize=(8,7), showPlot=True
    leverage = kwargs.get('leverage',0)
    equityCurveSavePath = kwargs.get('equityCurveSavePath',None)
    pngPath = kwargs.get('pngPath',None)
    pngFilename = kwargs.get('pngFilename',None)
    figsize = kwargs.get('figsize',(8,7))
    showPlot =kwargs.get('showPlot',True)
    verbose = kwargs.get('verbose',True)
    totalTrades= kwargs.get('totalTrades',None)
    totalAvgTrades=kwargs.get('totalAvgTrades',None)
    asset=kwargs.get('asset','FX')
    totalCommissions =kwargs.get('totalCommissions',None)
    initialEquity = 100000.0
    nrows = SST.gainAhead.shape[0]
    #signalCounts = SST.signals.shape[0]        
    equityCurves = {}
    for trade in ['l','s','b']:       
        trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
        commission = pd.Series(data=0.0, index=range(0,len(SST.index)), name='commission')
        numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
        equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
        maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
        drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
        maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
        safef = pd.Series(data=leverage,index=range(0,len(SST.index)),name='safef')

        for i in range(0,len(SST.index)):
            if i == 0:
                if SST.signals[i] != 0:
                    positionChg =  safef[i] * initialEquity
                    commission[i]=IBcommission(SST,i,positionChg, safef,asset)
                else:
                    positionChg=0
                    commission[i]=0
                    
                equity[i] = initialEquity-commission[i]
                trades[i] = 0.0
                numBars[i] = 0.0
                maxEquity[i] = initialEquity
                drawdown[i] = 0.0
                maxDD[i] = 0.0

            else:
                if trade=='l':
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]                       
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                        numBars[i] = numBars[i-1] + 1 
                        #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                    else:
                        trades[i] = 0.0
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]                      
                        numBars[i] = numBars[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                        
                elif trade=='s':
                    if (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]          
                        numBars[i] = numBars[i-1] + 1 
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]          
                        numBars[i] = numBars[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]                            
                        numBars[i] = numBars[i-1] + 1                
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    elif (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]          
                        numBars[i] = numBars[i-1] + 1                
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        equity[i] = equity[i-1] + trades[i]
                        positionChg = (SST.signals[i]*safef[i]-SST.signals[i-1]*safef[i-1])*equity[i]
                        commission[i]  = IBcommission(SST,i,positionChg, safef,asset)
                        equity[i] -= commission[i]          
                        numBars[i] = numBars[i-1]
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
            
        equityCurves[trade] = pd.concat([SSTcopy.reset_index(), trades, commission, numBars, equity,maxEquity,drawdown,maxDD], axis =1)

    #  Compute cumulative equity for all days (buy and hold)   
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    commission = pd.Series(data=0.0, index=range(0,len(SST.index)), name='commission')
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
            commission[i]=0
            numBars[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0
        else:
            trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
            commission[i]=0
            numBars[i] = numBars[i-1] + 1 
            equity[i] = equity[i-1] + trades[i]-commission[i]
            maxEquity[i] = max(equity[i],maxEquity[i-1])
            drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
            maxDD[i] = max(drawdown[i],maxDD[i-1])          
    SSTcopy.loc[SST.signals==-1,'signals']=1
    SSTcopy.loc[SST.signals==0,'signals']=1

    equityCurves['buyHold'] = pd.concat([SSTcopy.reset_index(), safef, trades, commission, numBars, equity,maxEquity,drawdown,maxDD], axis =1)

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
        #ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    else:
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d")
        #ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    
    
    minorLocator = MultipleLocator(SST.shape[0])
    ax1.xaxis.set_minor_locator(minorLocator)
    ax2.xaxis.set_minor_locator(minorLocator)
    ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    #ax2.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
    ax2.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
    # use a more precise date string for the x axis locations in the
    # toolbar

    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.set_title(title)
    ax1.set_ylabel("TWR")
    ax1.legend(loc="best")
    ax2.set_ylabel("Drawdown")
    #gets rid of space at the end
    ax1.set_xlim(0, SST.shape[0])
    ax2.set_xlim(0, SST.shape[0])
    #shows last date index
    xticks = ax1.xaxis.get_minor_ticks()
    xticks[1].label1.set_visible(False)
    xticks = ax2.xaxis.get_minor_ticks()
    xticks[1].label1.set_visible(False)
    
    if totalTrades != None and totalAvgTrades != None and totalCommissions != None:
        text=  '\n%0.f signal counts: ' % nrows
        text+='\nAverage trades per hour per system: %0.2f' \
                                % (totalAvgTrades)
        text+='\nTWR for %i beLong and beShort trades with DPS is %0.4f, maxDD %0.4f' %\
                    (totalTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
        text+='\nTotal Commissions: $%0.2f' \
                                % (totalCommissions)
        plt.figtext(0.05,-0.05,text, fontsize=15)
    else:
        text=  '\n%0.f signal counts: ' % nrows
        if 1 in SST.signals.value_counts():
            text+= '%i beLong,  ' % SST.signals.value_counts()[1]
        if -1 in SST.signals.value_counts():
            text+= '%i beShort,  ' % SST.signals.value_counts()[-1]
        if 0 in SST.signals.value_counts():
            text+= '%i beFlat  ' % SST.signals.value_counts()[0]
        shortTrades, longTrades = numberZeros(SST.signals)
        allTrades = sum((SST.signals * SST.safef).round().diff().fillna(0).values !=0)
        hoursTraded = (SST.index[-1]-SST.index[0]).total_seconds()/60.0/60.0
        avgTrades = float(allTrades)/hoursTraded
        #text+='\nValidation Period from %s to %s' % (str(SST.index[0]), str(SST.index[-1]))
        text+='\nAverage trades per hour: %0.2f' % (avgTrades)
        text+=  '\nTWR for Buy & Hold is %0.4f, %i Bars, maxDD %0.4f' %\
                    (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
        text+='\nTWR for %i beLong trades is %0.4f, maxDD %0.4f' %\
                    (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
        text+='\nTWR for %i beShort trades is %0.4f, maxDD %0.4f' %\
                    (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
        text+='\nTWR for %i beLong and beShort trades with DPS is %0.4f, maxDD %0.4f' %\
                    (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
        text+='\nAverage SAFEf: %0.4f  Total Comm: $%0.2f' % (equityCurves['b'].safef.mean(),sum(equityCurves['b'].commission))
        plt.figtext(0.05,-0.15,text, fontsize=15)
    
    
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    #if pngPath != None:
    #    plt.savefig(pngPath+title+'.png', bbox_inches='tight')
    if pngPath != None and pngFilename != None:
        
        spstr=pngFilename.split('_')
        #print pngFilename, spstr
        if len(spstr) == 3:
            (ver, inst, mins)=spstr
            #print '.' in ver
            if '.' in ver:
                (ver1, ver2)=ver.split('.')
            else:
                ver1 =ver
            pngFilename = ver1 + '_' + inst
            
        print 'Saving: ' + pngPath+pngFilename+'.png'
        plt.savefig(pngPath+pngFilename+'.png', bbox_inches='tight')
        
    if showPlot:
        plt.show()
    plt.close()
    
    '''
    if verbose:
        print '\nThere are %0.f signal counts' % nrows
        if 1 in SST.signals.value_counts():
            print SST.signals.value_counts()[1], 'beLong Signals',
        if -1 in SST.signals.value_counts():
            print SST.signals.value_counts()[-1], 'beShort Signals',
        if 0 in SST.signals.value_counts():
            print SST.signals.value_counts()[0], 'beFlat Signals',
            
    if verbose:
        hoursTraded = (SST.index[-1]-SST.index[0]).total_seconds()/60.0/60.0
        avgTrades = float(allTrades)/hoursTraded
        print 'Validation Period from %s to %s' % (str(SST.index[0]), str(SST.index[-1]))
        print 'Average trades per hour: %0.2f' % (avgTrades)
        print 'TWR for Buy & Hold is %0.3f, %i Bars, maxDD %0.3f' %\
                    (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
        print 'TWR for %i beLong trades is %0.3f, maxDD %0.3f' %\
                    (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
        print 'TWR for %i beShort trades is %0.3f, maxDD %0.3f' %\
                    (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
        print 'TWR for %i beLong and beShort trades with DPS is %0.3f, maxDD %0.3f' %\
                    (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
        print 'Average SAFEf: %0.3f,' % (equityCurves['b'].safef.mean())
    '''
    SST_equity = equityCurves['b']
    if 'dates' in SST_equity:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['dates'])).drop(['dates'], axis=1)
    else:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['index'])).drop(['index'], axis=1)
        

    
signalFiles = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
dataFiles = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
#contracts = [x.replace(barSize,'').replace('.csv','').replace('_','') for x in dataFiles if barSize in x]
dataSets = {}
maxCTs = {}
numTrades = {}
comms = {}
for contract in futures:
    #dataFilename = [f for f in dataFiles if contract in f][0]
    #dataFile = pd.read_csv(str(dataPath) + str(dataFilename), index_col='Date').drop_duplicates()
    dataFile = pd.read_csv(dataPath+'F_'+contract+'.txt', index_col=0)
    #dataFile = dataFile.drop([' P',' R', ' RINFO'],axis=1)
    dataFile.index = pd.to_datetime(dataFile.index,format='%Y%m%d')
    #dataFile.columns = ['Open','High','Low','Close','Volume','OI']
    print 'Loaded data from', str(dataPath) + str('F_'+contract)
    
    validSignalFiles = {}
    for version in versions: 
        validSignalFiles[version]=[f for f in signalFiles if version in f and barSize in f]
    #for f in [sfile for sfilelist in validSignalFiles for sfile in sfilelist]:
    for version in validSignalFiles:
        if version == 'v1.3':
            for f in [sfile for sfile in validSignalFiles[version] if contract in sfile]:
	      filename=str(signalPath)+str(f)
	      #print filename
	      if os.path.isfile(filename):
                signalFile = pd.read_csv(filename, index_col='dates').drop_duplicates()
                print 'Loaded signals from', filename
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
                    dataSet = dataSet[['signals','gainAhead','safef']][-lookback:].dropna()
                    maxCT = max(reindexed_sst.cycleTime.fillna(0).round())
                    title = version +'_maxLag_'+str(maxCT)+\
                                ' '+ reindexed_sst.iloc[-1].system          
                    equityCurve = calcEquity_signals(dataSet,title,\
                                        leverage = dataSet.safef.values, equityCurveSavePath=equityCurveSavePath,\
                                        pngPath=pngPath,verbose=verbose, pngFilename=f[:-4],\
                                        figsize=size, showPlot=showPlot)
                    csvFile=f[:-4]
                    spstr=csvFile.split('_')
        
                    if len(spstr) == 3:
                        (ver, inst, mins)=spstr
                        (ver1, ver2)=ver.split('.')
                        csvFile = ver1 + '_' + inst
                        
                    print "Saving: " + equityCurveSavePath+csvFile +'.csv'
                    equityCurve.to_csv(equityCurveSavePath+csvFile+'.csv')
                    
                    dataSets[title] = dataSet
                    maxCTs[title] = maxCT
                    numTrades[title] = sum((dataSet.signals * dataSet.safef).round().diff().fillna(0).values !=0)
                    comms[title] = equityCurve.commission.sum()
        else:
            for f in [sfile for sfile in validSignalFiles[version] if contract in sfile]:
                signalFile = pd.read_csv(str(signalPath)+str(f), index_col='dates').drop_duplicates()
                signalFile.index = pd.to_datetime(signalFile.index,format='%Y-%m-%d')
                print 'Loaded signals from', f
                #if there is no prior index in the row, then it's a legacy file
                #if 'prior_index' in signalFile:
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
                dataSet = dataSet[['signals','gainAhead','safef']][-lookback:].dropna()
                maxCT = max(reindexed_sst.cycleTime.fillna(0).round())
                title = f[:-4] +'_maxLag_'+str(maxCT)+\
                            ' '+ reindexed_sst.iloc[-1][version+'_system'] 
                equityCurve = calcEquity_signals(dataSet, title,\
                                    leverage = dataSet.safef.values,\
                                    equityCurveSavePath=equityCurveSavePath,\
                                    figsize=size, showPlot=showPlot,\
                                    pngPath=pngPath, pngFilename=f[:-4],\
                                    verbose=verbose)
                csvFile=f[:-4]
                spstr=csvFile.split('_')
    
                if len(spstr) == 3:
                    (ver, inst, mins)=spstr
                    if '.' in ver:
                        (ver1, ver2)=ver.split('.')
                    else:
                        ver1 =ver
                    #(ver1, ver2)=ver.split('.')
                    csvFile = ver1 + '_' + inst
                    
                print "Saving: " + equityCurveSavePath+csvFile +'.csv'
                equityCurve.to_csv(equityCurveSavePath+csvFile+'.csv')
                
                dataSets[title] = dataSet
                maxCTs[title] = maxCT
                numTrades[title] = sum((dataSet.signals * dataSet.safef).round().diff().fillna(0).values !=0)
                comms[title] = equityCurve.commission.sum()

#set arbitrary start/enddates that exist
startDate = dataSet.index[0]
endDate = dataSet.index[-1]

for title in dataSets:
    if dataSets[title].index[0]>startDate:
        startDate =  dataSets[title].index[0]
        
    if  dataSets[title].index[-1]<endDate:
        endDate =  dataSets[title].index[-1]


#title by version
titleDict = {}
for ver in versions:                    
    titleDict[ver] = [title for title in dataSets if str(ver) in title]

#create equal length equity curves.
eCurves = {}
for title in dataSets:
    eCurves[title] = calcEquity_signals(dataSets[title][startDate:endDate],title,\
                                        leverage = dataSets[title][startDate:endDate].safef.values,\
                                        equityCurveSavePath=None,\
                                        pngPath=None, verbose=False,\
                                        figsize=size, showPlot=False)
                                        
#find max cycletime and date index by version
eCurves_bySystem = {}
dateIndexes = {}
maxCT_bySystem = {}
totalTrades_bySystem = {}
totalComms_bySystem = {}
for ver in titleDict:
    for i,title in enumerate(titleDict[ver]):
        if i == 0:
            dateIndex = eCurves[title].index
            maxCT = maxCTs[title]
            trades = numTrades[title]
            totalComms = comms[title]
        else:
            dateIndex = dateIndex.intersection(eCurves[title].index)
            trades += numTrades[title]
            totalComms+=comms[title]
            if maxCTs[title] > maxCT:
                maxCT = maxCTs[title]
    dateIndexes[ver] = dateIndex
    maxCT_bySystem[ver] = maxCT
    totalTrades_bySystem[ver] = trades
    totalComms_bySystem[ver]=totalComms
    
#find max cycletime and date index of all versions
all = str(versions).replace('[','').replace('\'','').replace(']','')
for i,title in enumerate(eCurves):
    if i == 0:
        dateIndex = eCurves[title].index
        maxCT = maxCTs[title]
        trades = numTrades[title]
        totalComms = comms[title]
    else:
        dateIndex = dateIndex.intersection(eCurves[title].index)
        trades += numTrades[title]
        totalComms+=comms[title]
        if maxCTs[title] > maxCT:
            maxCT = maxCTs[title]
    dateIndexes[all] = dateIndex
    maxCT_bySystem[all] = maxCT
    totalTrades_bySystem[all] = trades
    totalComms_bySystem[all]=totalComms
    
#add by version
for ver in titleDict:
    equityBySystem = np.zeros(dateIndexes[ver].shape[0])
    for title in titleDict[ver]:
        equityBySystem += eCurves[title].ix[dateIndexes[ver]].equity.values
    eCurves_bySystem[str(len(titleDict[ver]))+' '+ver+' Systems'] = equityBySystem
    
# add everything together
equityCons = np.zeros(dateIndexes[all].shape[0])
for title in dataSets:
    equityCons += eCurves[title].ix[dateIndexes[all]].equity.values
eCurves_bySystem[str(len(dataSets))+' '+all+' Systems'] = equityCons

#create price change
for title in eCurves_bySystem:
    dI_title = title[len(title.split()[0])+1:-8]
    equityCons = pd.concat([pd.Series(data=priceChange(eCurves_bySystem[title]),\
                                        name = 'gainAhead', index = dateIndexes[dI_title]),\
                            pd.Series(data=1, name='safef', index = dateIndexes[dI_title]),\
                            pd.Series(data=1, name='signals', index = dateIndexes[dI_title])],\
                            axis=1)
    #add cycletime
    hoursTraded = (dateIndexes[dI_title][-1]-dateIndexes[dI_title][0]).total_seconds()/60.0/60.0
    avgTrades = float(totalTrades_bySystem[dI_title])/float(title.split()[0])/hoursTraded
    title2 = title+' maxLag'+str(maxCT_bySystem[dI_title])
    #print '\n\n'
    eCurves[title2] = calcEquity_signals(equityCons,title2,\
                                    leverage = equityCons.safef.values,\
                                    equityCurveSavePath=equityCurveSavePath,\
                                    pngPath=pngPath, verbose=False, pngFilename=dI_title,\
                                    figsize=size, showPlot=showPlot, 
                                    totalTrades= totalTrades_bySystem[dI_title],
                                    totalAvgTrades=avgTrades,
                                    totalCommissions =  totalComms_bySystem[dI_title])
                                    
    spstr=dI_title.split('_')
    if len(spstr) == 3:
        (ver, inst, mins)=spstr
        (ver1, ver2)=ver.split('.')
        dI_title = ver1 + '_' + inst
    print 'Saving: ' + equityCurveSavePath+dI_title+'.csv'    
    eCurves[title2].to_csv(equityCurveSavePath+dI_title+'.csv')
    
    #print 'Validation Period from %s to %s' % (str(equityCons.index[0]), str(equityCons.index[-1]))
    #print 'There are %0.f bars. Average trades per hour per system: %0.2f' \
    #                       % (equityCons.shape[0], avgTrades)
    #print 'TWR for %i beLong and beShort trades with DPS is %0.5f, maxDD %0.5f' %\
    #           (totalTrades_bySystem[dI_title], eCurves[title2].equity.iloc[-1],\
    #               eCurves[title2].maxDD.iloc[-1])
    #eCurves[title2].to_csv(equityCurveSavePath+dI_title+'_curve.csv')


#check
#for x in eCurves_bySystem:
#    print x
#    print eCurves_bySystem[x][-10:]
#    print priceChange(eCurves_bySystem[x][-10:])
#    print dateIndexes[x].shape    

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
            
            

