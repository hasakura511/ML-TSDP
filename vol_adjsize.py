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

tradingEquity=1000000
riskPerTrade=0.01
riskEquity=tradingEquity*riskPerTrade
lookback=2
refresh=False

if len(sys.argv)==1:
    dataPath='D:/ML-TSDP/data/csidata/v4futures/'
    savePath='D:/ML-TSDP/data/'
    signalPath = 'D:/ML-TSDP/data/signals/' 
else:
    dataPath='./data/csidata/v4futures/'
    savePath='./data/'
    signalPath = './data/signals/' 
    
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
marketList = [x.split('_')[0] for x in files]
  
futuresDF=pd.DataFrame()
for sym in marketList:
    data = pd.read_csv(dataPath+sym+'_B.csv')[-lookback-1:]
    data.columns=['Date','Open','High','Low','Close','Vol','OI','R']
    atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
    #print sym, atr,data.tail()
    futuresDF.set_value(sym,data.Date.iloc[-1],atr[-1])
futuresDF=futuresDF.sort_index()
print futuresDF
futuresDF.to_csv(savePath+'futuresATR.csv')

signalDF=pd.DataFrame()
for contract in marketList:
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='v4_'+sym+'.csv'
    print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0, header=None)
    signalDF.set_value(sym,data.Date.iloc[-1],data.signals.iloc[-1])
    
signalDF=signalDF.sort_index()
print signalDF
signalDF.to_csv(savePath+'futuresSignals.csv')
