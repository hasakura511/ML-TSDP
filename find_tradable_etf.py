# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:12:20 2016

@author: Hidemi
"""


import datetime
import pandas as pd
import numpy as np
import requests
from pandas.io.data import DataReader
#from suztoolz.loops import calcEquity_df
from suztoolz.transform import gainAhead, numberZeros
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime as dt

start = '01/01/1999'
end = '03/22/2016'
rsPath = 'C:/Users/Hidemi/Desktop/Python/data/from_RS/'
rsfile = '0_DoubleDF_ES_RegimeSwitching_wl2.0_CAR25.csv'

def getDailyGoogle(ticker, start, end):
    #ticker = 'SDS'
    source = 'google'
    #start = '01/01/1999'
    #end = '03/22/2016'
    qt = DataReader(ticker, source, start, end)
    print qt.head()
    print qt.tail()
    return qt

def calcEquity(SST, title, benchmark):
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
        num_days = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numDays')
        equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
        maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
        drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
        maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
        safef = pd.Series(data=SST.safef.values,index=range(0,len(SST.index)),name='safef')

        for i in range(0,len(SST.index)):
            if i == 0:
                equity[i] = initialEquity
                trades[i] = 0.0
                num_days[i] = 0.0
                maxEquity[i] = initialEquity
                drawdown[i] = 0.0
                maxDD[i] = 0.0

            else:
                if trade=='l':
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        num_days[i] = num_days[i-1] + 1 
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])

                        #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                    else:
                        trades[i] = 0.0
                        num_days[i] = num_days[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                elif trade=='s':
                    if (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        num_days[i] = num_days[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        num_days[i] = num_days[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        num_days[i] = num_days[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    elif (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        num_days[i] = num_days[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        num_days[i] = num_days[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                        
        SSTcopy = SST.copy(deep=True)
        if trade =='l':
            #changeIndex = SSTcopy.signals[SST.signals==-1].index
            SSTcopy.signals[SST.signals==-1]=0
        elif trade =='s':
            #changeIndex = SSTcopy.signals[SST.signals==1].index
            SSTcopy.signals[SST.signals==1]=0
            
        equityCurves[trade] = pd.concat([SSTcopy.reset_index(), safef, trades, num_days, equity,maxEquity,drawdown,maxDD], axis =1)

    #  Compute cumulative equity for all days (buy and hold)   
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    num_days = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numDays')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    safef = pd.Series(data=1.0,index=range(0,len(SST.index)),name='safef')
    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            num_days[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0
        else:
            trades[i] = safef[i-1] * equity[i-1] * benchmark.gainAhead[i-1]
            num_days[i] = num_days[i-1] + 1 
            equity[i] = equity[i-1] + trades[i]
            maxEquity[i] = max(equity[i],maxEquity[i-1])
            drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
            maxDD[i] = max(drawdown[i],maxDD[i-1])       
    SSTcopy = benchmark.copy(deep=True)        
    SSTcopy.signals[SSTcopy.signals==-1]=1
    SSTcopy.signals[SSTcopy.signals==0]=1       
    equityCurves['buyHold'] = pd.concat([SSTcopy.reset_index()[['signals','gainAhead']], safef, trades, num_days, equity,maxEquity,drawdown,maxDD], axis =1)

        
    #plt.close('all')
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(8,7))
    #plt.subplot(2,1,1)
    ax1.plot(SST.index.to_datetime(), equityCurves['l'].equity,label="Long 1 Signals",color='b')
    ax1.plot(SST.index.to_datetime(), equityCurves['s'].equity,label="Short -1 Signals",color='r')
    ax1.plot(SST.index.to_datetime(), equityCurves['b'].equity,label="Long & Short",color='g')
    ax1.plot(SST.index.to_datetime(), equityCurves['buyHold'].equity,label="BuyHold",ls='--',color='c')
    #fig, ax = plt.subplots(2)
    #plt.subplot(2,1,2)
    ax2.plot(SST.index.to_datetime(), -equityCurves['l'].drawdown,label="Long 1 Signals",color='b')
    ax2.plot(SST.index.to_datetime(), -equityCurves['s'].drawdown,label="Short -1 Signals",color='r')
    ax2.plot(SST.index.to_datetime(), -equityCurves['b'].drawdown,label="Long & Short",color='g')
    ax2.plot(SST.index.to_datetime(), -equityCurves['buyHold'].drawdown,label="BuyHold",ls='--',color='c')
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the
    # toolbar

    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.set_title(title)
    ax1.set_ylabel("TWR")
    ax1.legend(loc="best")
    ax2.set_ylabel("Drawdown")   
    plt.show()
    shortTrades, longTrades = numberZeros(SST.signals)
    allTrades = shortTrades+ longTrades
    print '\nValidation Period from', SST.index[0],'to',SST.index[-1]
    print 'TWR for Buy & Hold is %0.3f, %i days, maxDD %0.3f' %\
                (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
    print 'TWR for %i beLong trades is %0.3f, maxDD %0.3f' %\
                (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
    print 'TWR for %i beShort trades is %0.3f, maxDD %0.3f' %\
                (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
    print 'TWR for %i beLong and beShort trades is %0.3f, maxDD %0.3f' %\
                (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
    #print 'avg SAFEf:', leverage

    return equityCurves['b']

    
sso = getDailyGoogle('SSO', start,end)
sds = getDailyGoogle('SDS',start,end)
spy = getDailyGoogle('SPY',start,end)

sst = pd.read_csv(rsPath+rsfile,index_col='Unnamed: 0')
sst.index = sst.index.to_datetime()

sso_ga = pd.Series(data=gainAhead(sso.Close), index=sso.index, name='sso_gA')
sso_close = pd.Series(data=sso.Close.values, index=sso.index, name='sso_close')
sds_ga = pd.Series(data=gainAhead(sds.Close), index=sds.index, name='sds_gA')
sds_close = pd.Series(data=sds.Close.values, index=sds.index, name='sds_close')
spy_ga = pd.Series(data=gainAhead(spy.Close), index=spy.index, name='gainAhead')
spy_close = pd.Series(data=spy.Close.values, index=spy.index, name='spy_close')

sst2 = pd.concat([sst[['signals','safef','gainAhead']],sso_ga,sso_close,\
                                                sds_ga,sds_close],axis=1).dropna()
ga2 = pd.Series(data=0, index=sst2.index, name='gainAhead', dtype='float')
safef2 = pd.Series(data=0, index=sst2.index, name='safef', dtype='float')
for x in sst2.index:
    if sst2.ix[x].signals == 1:
        #2x leverage so safef needs to be 1
        ga2[x] = sst2.ix[x].sso_gA
        safef2[x] = sst2.ix[x].safef/2
        #print sst2.ix[x].signals, sst2.ix[x].sds_gA,ga2[x]
    elif sst2.ix[x].signals == -1:
        
        ga2[x] = -sst2.ix[x].sds_gA
        safef2[x] = sst2.ix[x].safef/2
        #print sst2.ix[x].signals, sst2.ix[x].sds_gA,ga2[x]
    else:
        ga2[x] = 0
        safef2[x] = 0
        #print sst2.ix[x].signals, ga2[x]
        
SST = pd.concat([sst2.signals, safef2, ga2],axis=1)
title = 'SSO/SDS'
SSOSDS =calcEquity(SST, title, sst2)

SST2 = pd.concat([sst2.signals, sst2.safef, spy_ga],axis=1).dropna()
title = 'SPY'
SPY = calcEquity(SST2, title, sst2)
