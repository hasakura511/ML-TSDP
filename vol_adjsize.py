# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import time
import math
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
from datetime import datetime as dt
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
riskEquity=2000
lookback=20
refresh=False
currencyFile = 'currenciesATR.csv'
systemFilename='system_v4futures.csv'
safefAdjustment=0.25

if len(sys.argv)==1:
    showPlots=False
    dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    signalPath = 'D:/ML-TSDP/data/signals/' 
    systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
    
else:
    showPlots=False
    dataPath='./data/csidata/v4futures2/'
    savePath='./data/'
    signalPath = './data/signals/' 
    savePath2 = './data/results/'
    systemPath =  './data/systems/'

fxRates=pd.read_csv(savePath+currencyFile, index_col=0)
for i,col in enumerate(fxRates.columns):
    if 'Close' in col:
        fxRates = fxRates[fxRates.columns[i]]
        break
        
offline = ['AC','CGB','EBS','ED','FEI','FSS','YB']

fxDict={
    'AUD':1/fxRates.ix['AUDUSD'],
    'CAD':fxRates.ix['USDCAD'],
    'CHF':fxRates.ix['USDCHF'],
    'EUR':1/fxRates.ix['EURUSD'],
    'GBP':1/fxRates.ix['GBPUSD'],
    'HKD':7.77,
    'JPY':fxRates.ix['USDJPY'],
    'NZD':1/fxRates.ix['NZDUSD'],
    'SGD':1.34,
    'USD':1,
    }

c2contractSpec = {
    'AC':['@AC',fxDict['USD'],29000],
    'AD':['@AD',fxDict['USD'],100000],
    'AEX':['AEX',fxDict['EUR'],200],
    'BO':['@BO',fxDict['USD'],600],
    'BP':['@BP',fxDict['USD'],62500],
    'C':['@C',fxDict['USD'],50],
    'CC':['@CC',fxDict['USD'],10],
    'CD':['@CD',fxDict['USD'],100000],
    'CGB':['CB',fxDict['CAD'],1000],
    'CL':['QCL',fxDict['USD'],1000],
    'CT':['@CT',fxDict['USD'],500],
    'CU':['@EU',fxDict['USD'],125000],
    'DX':['@DX',fxDict['USD'],1000],
    'EBL':['BD',fxDict['EUR'],1000],
    'EBM':['BL',fxDict['EUR'],1000],
    'EBS':['EZ',fxDict['EUR'],1000],
    'ED':['@ED',fxDict['USD'],2500],
    'EMD':['@EMD',fxDict['USD'],100],
    'ES':['@ES',fxDict['USD'],50],
    'FC':['@GF',fxDict['USD'],500],
    'FCH':['MT',fxDict['EUR'],10],
    'FDX':['DXM',fxDict['EUR'],5],
    'FEI':['IE',fxDict['EUR'],2500],
    'FFI':['LF',fxDict['GBP'],10],
    'FLG':['LG',fxDict['GBP'],1000],
    'FSS':['LL',fxDict['GBP'],1250],
    'FV':['@FV',fxDict['USD'],1000],
    'GC':['QGC',fxDict['USD'],100],
    'HCM':['HHI',fxDict['HKD'],50],
    'HG':['QHG',fxDict['USD'],250],
    'HIC':['HSI',fxDict['HKD'],50],
    'HO':['QHO',fxDict['USD'],42000],
    'JY':['@JY',fxDict['USD'],125000],
    'KC':['@KC',fxDict['USD'],375],
    'KW':['@KW',fxDict['USD'],50],
    'LB':['@LB',fxDict['USD'],110],
    'LC':['@LE',fxDict['USD'],400],
    'LCO':['EB',fxDict['USD'],1000],
    'LGO':['GAS',fxDict['USD'],100],
    'LH':['@HE',fxDict['USD'],400],
    'LRC':['LRC',fxDict['USD'],10],
    'LSU':['QW',fxDict['USD'],50],
    'MEM':['@MME',fxDict['USD'],50],
    'MFX':['IB',fxDict['EUR'],10],
    'MP':['@PX',fxDict['USD'],500000],
    'MW':['@MW',fxDict['USD'],50],
    'NE':['@NE',fxDict['USD'],100000],
    'NG':['QNG',fxDict['USD'],10000],
    'NIY':['@NKD',fxDict['JPY'],500],
    'NQ':['@NQ',fxDict['USD'],20],
    'O':['@O',fxDict['USD'],50],
    'OJ':['@OJ',fxDict['USD'],150],
    'PA':['QPA',fxDict['USD'],100],
    'PL':['QPL',fxDict['USD'],50],
    'RB':['QRB',fxDict['USD'],42000],
    'RR':['@RR',fxDict['USD'],2000],
    'RS':['@RS',fxDict['CAD'],20],
    'S':['@S',fxDict['USD'],50],
    'SB':['@SB',fxDict['USD'],1120],
    'SF':['@SF',fxDict['USD'],125000],
    'SI':['QSI',fxDict['USD'],50],
    'SIN':['IN',fxDict['USD'],2],
    'SJB':['BB',fxDict['JPY'],100000],
    'SM':['@SM',fxDict['USD'],100],
    'SMI':['SW',fxDict['CHF'],10],
    'SSG':['SS',fxDict['SGD'],200],
    'STW':['TW',fxDict['USD'],100],
    'SXE':['EX',fxDict['EUR'],10],
    'TF':['@TFS',fxDict['USD'],100],
    'TU':['@TU',fxDict['USD'],2000],
    'TY':['@TY',fxDict['USD'],1000],
    'US':['@US',fxDict['USD'],1000],
    'VX':['@VX',fxDict['USD'],1000],
    'W':['@W',fxDict['USD'],50],
    'YA':['AP',fxDict['AUD'],25],
    'YB':['HBS',fxDict['AUD'],2400],
    'YM':['@YM',fxDict['USD'],5],
    'YT2':['HTS',fxDict['AUD'],2800],
    'YT3':['HXS',fxDict['AUD'],8000],
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
    contractName=c2contractSpec[sym][0]+months[int(contractMonth)]+contractYear
    usdATR = atr[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
    qty = int(math.ceil(riskEquity/usdATR))
    #print sym, data.R[-1], contractName
    #signalFilename='v4_'+sym+'.csv'
    corrDF[sym]=pc
    futuresDF.set_value(sym,'Contract',contractName)
    futuresDF.set_value(sym,'LastClose',data.Close[-1])
    futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
    futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
    futuresDF.set_value(sym,'ACT',priorSig)
    futuresDF.set_value(sym,'usdATR',usdATR)
    futuresDF.set_value(sym,'QTY',qty)
    futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
    
    

    
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
    if sym in offline:
        adjQty=0
    else:
        if data.dpsSafef.iloc[-1] ==1:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
        else:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
        
    futuresDF.set_value(sym,'LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'LastSAFEf',data.dpsSafef.iloc[-1])
    futuresDF.set_value(sym,'finalQTY',adjQty)
    futuresDF.set_value(sym,'SIG'+str(data.index[-1]),data.signals.iloc[-1])
   

for i,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='0.5_'+sym+'_1D.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    if sym in offline:
        adjQty=0
    else:
        if data.dpsSafef.iloc[-1] ==1:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
        else:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
            
    futuresDF.set_value(sym,'0.5LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'0.5LastSAFEf',data.dpsSafef.iloc[-1])
    futuresDF.set_value(sym,'0.5finalQTY',adjQty)
    futuresDF.set_value(sym,'0.5SIG'+str(data.index[-1]),data.signals.iloc[-1])

for i,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='1_'+sym+'_1D.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    if sym in offline:
        adjQty=0
    else:
        if data.dpsSafef.iloc[-1] ==1:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
        else:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
            
    futuresDF.set_value(sym,'1LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'1LastSAFEf',data.dpsSafef.iloc[-1])
    futuresDF.set_value(sym,'1finalQTY',adjQty)
    futuresDF.set_value(sym,'1SIG'+str(data.index[-1]),data.signals.iloc[-1])
    
futuresDF=futuresDF.sort_index()
print futuresDF
print 'Saving', savePath+'futuresATR.csv'
futuresDF.to_csv(savePath+'futuresATR.csv')
futuresDF.to_csv(savePath2+'futuresATR_'+dt.now().strftime("%Y%m%d%H%M")+'.csv')

#system file update
system = pd.read_csv(systemPath+systemFilename)

for sys in system.System:
    sym=sys.split('_')[1]
    idx=system[system.System==sys].index[0]
    print sys, sym, system.ix[idx].c2qty,
    system.set_value(idx,'c2qty',int(futuresDF.ix[sym]['finalQTY']))
    print system.ix[idx].c2qty, system.ix[idx].c2sym,
    system.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
    print system.ix[idx].c2sym
    
print 'Saving', systemPath+systemFilename
system.to_csv(systemPath+systemFilename, index=False)
    
#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()