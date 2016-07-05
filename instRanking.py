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
groups  = {
        'AC':'energy',
        'AD':'currency',
        'AEX':'index',
        'BO':'grain',
        'BP':'currency',
        'C':'grain',
        'CC':'soft',
        'CD':'currency',
        'CGB':'rates',
        'CL':'energy',
        'CT':'soft',
        'CU':'currency',
        'DX':'currency',
        'EBL':'rates',
        'EBM':'rates',
        'EBS':'rates',
        'ED':'rates',
        'EMD':'index',
        'ES':'index',
        'FC':'meat',
        'FCH':'index',
        'FDX':'index',
        'FEI':'rates',
        'FFI':'index',
        'FLG':'rates',
        'FSS':'rates',
        'FV':'rates',
        'GC':'metal',
        'HCM':'index',
        'HG':'metal',
        'HIC':'index',
        'HO':'energy',
        'JY':'currency',
        'KC':'soft',
        'KW':'grain',
        'LB':'soft',
        'LC':'meat',
        'LCO':'energy',
        'LGO':'energy',
        'LH':'meat',
        'LRC':'soft',
        'LSU':'soft',
        'MEM':'index',
        'MFX':'index',
        'MP':'currency',
        'MW':'grain',
        'NE':'currency',
        'NG':'energy',
        'NIY':'index',
        'NQ':'index',
        'O':'grain',
        'OJ':'soft',
        'PA':'metal',
        'PL':'metal',
        'RB':'energy',
        'RR':'grain',
        'RS':'grain',
        'S':'grain',
        'SB':'soft',
        'SF':'currency',
        'SI':'metal',
        'SIN':'index',
        'SJB':'rates',
        'SM':'grain',
        'SMI':'index',
        'SSG':'index',
        'STW':'index',
        'SXE':'index',
        'TF':'index',
        'TU':'rates',
        'TY':'rates',
        'US':'rates',
        'VX':'index',
        'W':'grain',
        'YA':'index',
        'YB':'rates',
        'YM':'index',
        'YT2':'rates',
        'YT3':'rates',
        }
        
if len(sys.argv) > 1:
    #bestParamsPath = './data/params/'
    #signalPath = './data/signals/'
    #dataPath = './data/csidata/v4futures2/'
    equityCurveSavePath = './data/signalPlots/'
    savePath= './data/results/'
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
    #pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #showPlot = True
    #verbose = True

dataFiles = [ f for f in listdir(equityCurveSavePath) if \
                    isfile(join(equityCurveSavePath,f)) and '.csv' in f and version+'_' in f and\
                    len(f.split('_')[1].split('.')[0])<6]
print len(dataFiles), dataFiles

futuresRank=pd.DataFrame()
for f in dataFiles:
    lookback=10
    sym = f.split('_')[1].split('.')[0]
    df = pd.read_csv(equityCurveSavePath+f, index_col=0)
    df.index.name='dates'
    #print df
    nrows=df.shape[0]
    if nrows <lookback:
        lookback= nrows
        
    pctChg=(df.equity[-1]-df.equity[-lookback])/df.equity[-lookback]
    futuresRank.set_value(sym,'G/L Last '+str(lookback),pctChg)
    futuresRank.set_value(sym,'group',groups[sym])
    futuresRank.set_value(sym,str(df.index[-lookback])+' to '+str(df.index[-1]),pctChg)
futuresRank.index.name='Symbol'
futuresRank = futuresRank.sort_values(by=futuresRank.columns[0], ascending=False).reset_index()
print 'Saving', savePath+'fRank.html'
futuresRank.to_html(savePath+'fRank.html')
print futuresRank