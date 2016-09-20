import time
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

filter=False
start_time = time.time()
version = 'v4'
systemFilename='system_v4mini.csv'
c2id=101533256
#offline = ['AC','CGB','EBS','ED','FEI','FSS','YB']
offline  = ['AC','AEX','CGB','DX','EBL','EBM','EBS','ED','EMD','ES','FC','FCH','FDX','FEI','FFI','FLG','FSS','HCM','HIC','HO','KC','KW','LB','LC','LCO','LH','LRC','LSU','MFX','MP','MW','NE','NIY','O','OJ','S','SF','SI','SM','SMI','SSG','SXE','TF','TU','TY','VX','YA','YB','YM','YT2']


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
    atrPath='./data/futuresATR.csv'
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
    atrPath='D:/ML-TSDP/data/futuresATR.csv'
    systemPath =  'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
    #pngPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signalPlots/' 
    #showPlot = True
    #verbose = True
    
#load symbol names less than 6 chars long
dataFiles = [ f for f in listdir(equityCurveSavePath) if \
                    isfile(join(equityCurveSavePath,f)) and '.csv' in f and version+'_' in f and\
                    len(f.split('_')[1].split('.')[0])<6]
print len(dataFiles), dataFiles


atrFile = pd.read_csv(atrPath, index_col=0)

futuresRank=pd.DataFrame()
for f in dataFiles:
    
    sym = f.split('_')[1].split('.')[0]
    if sym in groups:
        #+1 to make the minimum index -2
        lookback=int(abs(atrFile.ix[sym].LastSRUN))+1
        print sym,lookback,
        df = pd.read_csv(equityCurveSavePath+f, index_col=0)
        df.index.name='dates'
        #print df
        nrows=df.shape[0]
        if nrows <lookback:
            lookback= nrows
            
        pctChg=(df.equity[-1]-df.equity[-lookback])/df.equity[-lookback]
        futuresRank.set_value(sym,'G/L Last',pctChg)
        futuresRank.set_value(sym,'group',groups[sym])
        futuresRank.set_value(sym,'lookback',lookback)
        futuresRank.set_value(sym,'LastDate',df.index[-1])
        futuresRank.set_value(sym,str(lookback)+' '+str(df.index[-lookback])+' to '+str(df.index[-1]),pctChg)
futuresRank.index.name='Symbol'
#Ascending=False rank by best, True rank by worst.
futuresRank = futuresRank.sort_values(by=futuresRank.columns[0], ascending=True).reset_index()
print '\nSaving', savePath+'fRank.html'
futuresRank.to_html(savePath+'fRank.html')
print futuresRank.ix[:,0:5]

if filter:
    print "\nRanking filter is ON.."
    #filter off un/profitable contracts
    onlineSymbols=[]
    for i,g in enumerate(futuresRank.group.unique()):
        
        for sym in futuresRank[futuresRank.group ==g].Symbol.values:
            if sym not in offline:
                #print sym
                online= futuresRank[futuresRank.group ==g][futuresRank.Symbol==sym].iloc[:,0:5]
                onlineSymbols.append(online.Symbol.values[0])
                
                break
        print i+1, online
        
    print '\nOnline:', len(onlineSymbols), onlineSymbols
    #offlineSymbols = sorted(futuresRank[futuresRank['G/L Last']<0].Symbol.values)
    offlineSymbols = [x for x in futuresRank.Symbol.values if x not in onlineSymbols]
    print '\nOffline:', len(offlineSymbols), offlineSymbols
    print 'Total:', len(onlineSymbols) + len(offlineSymbols)
    system = pd.read_csv(systemPath+systemFilename)
    
    for sys in system.System:
        sym=sys.split('_')[1]
        #print sym
        if sym in offlineSymbols:
            idx=system[system.System==sys].index[0]
            print sys, sym, system.ix[idx].c2qty,
            system.set_value(idx,'c2qty',0)
            print system.ix[idx].c2qty
    system.Name=systemFilename.split('_')[1][:-4]
    system.c2id=c2id
    print 'Saving', systemPath+systemFilename
    system.to_csv(systemPath+systemFilename, index=False)
else:
    print "\nRanking filter is OFF.."
    print '\nOffline:', len(offline), offline
    online=[x for x in groups.keys() if x not in offline]
    print '\nOnline:', len(online), online
    print 'Total:', len(online) + len(offline)
    system = pd.read_csv(systemPath+systemFilename)
    
    for sys in system.System:
        sym=sys.split('_')[1]
        #print sym
        if sym in offline:
            idx=system[system.System==sys].index[0]
            print sys, sym, system.ix[idx].c2qty,
            system.set_value(idx,'c2qty',0)
            print system.ix[idx].c2qty
    system.Name=systemFilename.split('_')[1][:-4]
    system.c2id=c2id
    print 'Saving', systemPath+systemFilename
    system.to_csv(systemPath+systemFilename, index=False)
#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()