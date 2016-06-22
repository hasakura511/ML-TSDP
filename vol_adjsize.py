# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import time
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
from suztoolz.datatools.seasonalClass import seasonalClassifier
start_time = time.time()
version='v4'
tradingEquity=1000000
riskPerTrade=0.01
riskEquity=tradingEquity*riskPerTrade
lookback=20
refresh=False
c2contracts = {
                    'AC':'@AC',
                    'AD':'@AD',
                    'AEX':'AEX',
                    'BO':'@BO',
                    'BP':'@BP',
                    'C':'@C',
                    'CC':'@CC',
                    'CD':'@CD',
                    'CGB':'CB',
                    'CL':'QCL',
                    'CT':'@CT',
                    'CU':'@EU',
                    'DX':'@DX',
                    'EBL':'BD',
                    'EBM':'BL',
                    'EBS':'EZ',
                    'ED':'@ED',
                    'EMD':'@EMD',
                    'ES':'@ES',
                    'FC':'@GF',
                    'FCH':'MT',
                    'FDX':'DXM',
                    'FEI':'IE',
                    'FFI':'LF',
                    'FLG':'LG',
                    'FSS':'LL',
                    'FV':'@FV',
                    'GC':'QGC',
                    'HCM':'HHI',
                    'HG':'QHG',
                    'HIC':'HSI',
                    'HO':'QHO',
                    'JY':'@JY',
                    'KC':'@KC',
                    'KW':'@KW',
                    'LB':'@LB',
                    'LC':'@LE',
                    'LCO':'EB',
                    'LGO':'GAS',
                    'LH':'@HE',
                    'LRC':'LRC',
                    'LSU':'QW',
                    'MEM':'@MME',
                    'MFX':'IB',
                    'MP':'@PX',
                    'MW':'@MW',
                    'NE':'@NE',
                    'NG':'QNG',
                    'NIY':'@NKD',
                    'NQ':'@NQ',
                    'O':'@O',
                    'OJ':'@OJ',
                    'PA':'QPA',
                    'PL':'QPL',
                    'RB':'QRB',
                    'RR':'@RR',
                    'RS':'@RS',
                    'S':'@S',
                    'SB':'@SB',
                    'SF':'@SF',
                    'SI':'QSI',
                    'SIN':'IN',
                    'SJB':'BB',
                    'SM':'@SM',
                    'SMI':'SW',
                    'SSG':'SS',
                    'STW':'TW',
                    'SXE':'EX',
                    'TF':'@TFS',
                    'TU':'@TU',
                    'TY':'@TY',
                    'US':'@US',
                    'VX':'@VX',
                    'W':'@W',
                    'YA':'AP',
                    'YB':'HBS',
                    'YM':'@YM',
                    'YT2':'HTS',
                    'YT3':'HXS'
                    }
months = {
                1:'F',
                2:'G',
                3:'H',
                4:'J',
                5:'K',
                6:'M',
                7:'N',
                8:'Q',
                9:'U',
                10:'V',
                11:'X',
                12:'Z'
                }
if len(sys.argv)==1:
    showPlots=False
    dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    signalPath = 'D:/ML-TSDP/data/signals/' 
    
else:
    showPlots=False
    dataPath='./data/csidata/v4futures2/'
    savePath='./data/'
    signalPath = './data/signals/' 
    savePath2 = './data/results/'
    
    
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
marketList = [x.split('_')[0] for x in files]
  
futuresDF=pd.DataFrame()
corrDF=pd.DataFrame()
for i,contract in enumerate(marketList):
    data = pd.read_csv(dataPath+contract+'_B.csv', index_col=0, header=None)[-lookback-1:]
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
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
    contractYear=str(data.R[-1])[3]
    contractMonth=str(data.R[-1])[-2:]
    contractName=c2contracts[sym]+months[int(contractMonth)]+contractYear
    print sym, data.R[-1], contractName
    #signalFilename='v4_'+sym+'.csv'
    corrDF[sym]=pc
    futuresDF.set_value(sym,'LastClose',data.Close[-1])
    futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
    futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
    futuresDF.set_value(sym,'ACT',priorSig)
    futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
    futuresDF.set_value(sym,'Contract',contractName)
    

    
for i,contract in enumerate(marketList):
    print i+1,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    #seasonality
    seaBias, currRun, date,vStart = seasonalClassifier(sym, dataPath, savePath=savePath2+version+'_'+sym+'_MODE2',\
                                                debug=showPlots)
    futuresDF.set_value(sym,'vSTART',vStart)
    futuresDF.set_value(sym,'LastSEA',seaBias)
    futuresDF.set_value(sym,'SEA'+str(date),seaBias)
    futuresDF.set_value(sym,'LastSRUN',currRun)
    futuresDF.set_value(sym,'SRUN'+str(date),currRun)
    
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
    
if len(sys.argv)==1 and showPlots:
    #print data.index[0],'to',data.index[-1]
    plt.show()
plt.close()

for i,col in enumerate(corrDF):
    plt.figure(figsize=(8,10))
    corrDF[col].sort_values().plot.barh(color='r')
    plt.axvline(0, color='k')
    plt.title(col+' '+str(lookback)+' Day Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
    plt.xlim(-1,1)
    plt.xticks(np.arange(-1,1.25,.25))
    plt.grid(True)
    filename='v4_'+col+'_CORREL'+'.png'
    if savePath2 != None:
        print i+1,'Saving '+savePath2+filename
        plt.savefig(savePath2+filename, bbox_inches='tight')
    
    if len(sys.argv)==1 and showPlots:
        #print data.index[0],'to',data.index[-1]
        plt.show()
    plt.close()

for i,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='v4_'+sym+'.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    futuresDF.set_value(sym,'LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'LastSAFEf',data.dpsSafef.iloc[-1])
    futuresDF.set_value(sym,'SIG'+str(data.index[-1]),data.signals.iloc[-1])


futuresDF=futuresDF.sort_index()
print futuresDF
futuresDF.to_csv(savePath+'futuresATR.csv')

#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'