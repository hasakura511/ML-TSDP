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

version = 'v4'
qty={
    'AUDNZD':7,
    'GBPAUD':4,
    'AUDJPY':7,
    'AUDCHF':7,
    'AUDUSD':7,
    'AUDCAD':7,
    'NZDCAD':7,
    'CADCHF':6,
    'NZDCHF':7,
    'GBPNZD':4,
    'GBPCHF':4,
    'GBPUSD':4,
    'GBPJPY':4,
    'GBPCAD':4,
    'EURNZD':4,
    'EURAUD':4,
    'EURCAD':4,
    'EURJPY':4,
    'EURCHF':4,
    'EURGBP':4,
    'EURUSD':4,
    'CADJPY':6,
    'NZDJPY':7,
    'CHFJPY':5,
    'NZDUSD':7,
    'USDCHF':5,
    'USDCAD':5,
    'USDJPY':5,
    }
    
systemFilename='system_v4currencies.csv'
if len(sys.argv) > 1:
    #bestParamsPath = './data/params/'
    #signalPath = './data/signals/'
    #dataPath = './data/csidata/v4futures2/'
    equityCurveSavePath = './data/signalPlots/'
    savePath= './data/results/'
    systemPath =  './data/systems/'
    #pngPath = './data/results/'
    #showPlot = False
    #verbose = False
else:
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    #signalPath = 'D:/ML-TSDP/data/signals/' 
    #dataPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/from_IB/'
    #dataPath = 'D:/ML-TSDP/data/csidata/v4futures2/'
    #bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    equityCurveSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    savePath= 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    systemPath =  'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
    
    #pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #showPlot = True
    #verbose = True

dataFiles = [ f for f in listdir(equityCurveSavePath) if \
                    isfile(join(equityCurveSavePath,f)) and '.csv' in f and version+'_' in f and\
                    len(f.split('_')[1].split('.')[0])==6]
print len(dataFiles), dataFiles

fxRank=pd.DataFrame()
for f in dataFiles:
    lookback=3
    sym = f.split('_')[1].split('.')[0]
    df = pd.read_csv(equityCurveSavePath+f, index_col=0)
    df.index.name='dates'
    #print df
    nrows=df.shape[0]
    if nrows <lookback:
        lookback= nrows

    pctChg=(df.equity[-1]-df.equity[-lookback])/df.equity[-lookback]
    fxRank.set_value(sym,'G/L Last '+str(lookback),pctChg)
    fxRank.set_value(sym,str(df.index[-lookback])+' to '+str(df.index[-1]),pctChg)
fxRank.index.name='Symbol'
fxRank = fxRank.sort_values(by=fxRank.columns[0], ascending=False).reset_index()
print 'Saving', savePath+'fxRank.html'
fxRank.to_html(savePath+'fxRank.html')
print fxRank
offlineSymbols = fxRank[fxRank['G/L Last '+str(lookback)]<0].Symbol
for sym in offlineSymbols:
    qty[sym]=0
    
system = pd.read_csv(systemPath+systemFilename)

for sym in system.c2sym:
    idx=system[system.c2sym==sym].index[0]
    print sym, system.ix[idx].c2qty,
    system.set_value(idx,'c2qty',qty[sym])
    print system.ix[idx].c2qty

system.to_csv(systemPath+systemFilename, index=False)
    