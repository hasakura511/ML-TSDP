# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import io
import traceback
import json
import imp
import urllib
import urllib2
import webbrowser
import re
import datetime
import time
import inspect
import os
import os.path
import sys
import ssl
from copy import deepcopy
from suztoolz.transform import ATR2
import matplotlib.pyplot as plt
import seaborn as sns

tradingEquity=1000000
riskPerTrade=0.01
riskEquity=tradingEquity*riskPerTrade
lookback=20
refresh=False

if len(sys.argv)==1:
    dataPath='D:/ML-TSDP/data/csidata/v4futures/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    signalPath = 'D:/ML-TSDP/data/signals/' 
else:
    dataPath='./data/csidata/v4futures/'
    savePath='./data/'
    signalPath = './data/signals/' 
    savePath2 = './data/results/'
    
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
marketList = [x.split('_')[0] for x in files]
  
futuresDF=pd.DataFrame()
corrDF=pd.DataFrame()
for contract in marketList:
    data = pd.read_csv(dataPath+contract+'_B.csv', index_col=0, header=None)[-lookback-1:]
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    data.columns = ['Open','High','Low','Close','Volume','OI','R']
    data.index.name = 'Dates'
    atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
    pc=data.Close.pct_change()
    priorSig=np.where(pc<0,-1,1)[-1]
    #print pc
    #print sym, atr,data.tail()
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    #signalFilename='v4_'+sym+'.csv'
    corrDF[sym]=pc
    futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
    futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
    futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
    futuresDF.set_value(sym,'ACT'+str(data.index[-2]),priorSig)

#corrDF.to_csv(savePath+'futuresPCcsv')
#corrDF.corr().to_csv(savePath+'futuresCorr.csv')
corrDF=corrDF.corr()
fig,ax = plt.subplots(figsize=(13,13))
ax.set_title('Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
sns.heatmap(ax=ax,data=corrDF)
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
corrDF.to_html(savePath2+'futures_3.html')

if savePath != None:
    print 'Saving '+savePath2+'futures_3.png'
    fig.savefig(savePath2+'futures_3.png', bbox_inches='tight')
    
if len(sys.argv)==1:
    print data.index[0],'to',data.index[-1]
    plt.show()
plt.close()

for col in corrDF:
    plt.figure(figsize=(8,10))
    corrDF[col].sort_values().plot.barh(color='r')
    plt.axvline(0, color='k')
    plt.title(col+' Correlation'+str(lookback))
    plt.xlim(-1,1)
    plt.xticks(np.arange(-1,1.25,.25))
    plt.grid(True)
    filename='v4_'+col+'_CORREL'+'.png'
    if savePath2 != None:
        print 'Saving '+savePath2+filename
        plt.savefig(savePath2+filename, bbox_inches='tight')
    
    if len(sys.argv)==1:
        #print data.index[0],'to',data.index[-1]
        plt.show()
    plt.close()


    
for contract in marketList:
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='v4_'+sym+'.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    futuresDF.set_value(sym,'SIG'+str(data.index[-1]),data.signals.iloc[-1])

futuresDF=futuresDF.sort_index()
print futuresDF
futuresDF.to_csv(savePath+'futuresATR.csv')

#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')
