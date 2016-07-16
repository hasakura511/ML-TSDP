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
systemFilename2='system_v4mini.csv'
#range (-1 to 1) postive for counter-trend negative for trend i.e.
#-1 would 0 safef ==1 and double safef==2
#1 would 0 safef ==2 and double safef==1
safefAdjustment=0

if len(sys.argv)==1:
    showPlots=False
    dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
    dataPath2='D:/ML-TSDP/data/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    signalPath = 'D:/ML-TSDP/data/signals/' 
    signalSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
    
else:
    showPlots=False
    dataPath='./data/csidata/v4futures2/'
    dataPath2='./data/'
    savePath='./data/'
    signalPath = './data/signals/' 
    signalSavePath = './data/signals/' 
    savePath2 = './data/results/'
    systemPath =  './data/systems/'

fxRates=pd.read_csv(dataPath2+currencyFile, index_col=0)
for i,col in enumerate(fxRates.columns):
    if 'Last' in col:
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
#csisym:[c2sym,usdFXrate,multiplier,riskON signal]
c2contractSpec = {
    'AC':['@AC',fxDict['USD'],29000,1],
    'AD':['@AD',fxDict['USD'],100000,1],
    'AEX':['AEX',fxDict['EUR'],200,1],
    'BO':['@BO',fxDict['USD'],600,1],
    'BP':['@BP',fxDict['USD'],62500,1],
    'C':['@C',fxDict['USD'],50,1],
    'CC':['@CC',fxDict['USD'],10,1],
    'CD':['@CD',fxDict['USD'],100000,1],
    'CGB':['CB',fxDict['CAD'],1000,-1],
    'CL':['QCL',fxDict['USD'],1000,1],
    'CT':['@CT',fxDict['USD'],500,1],
    'CU':['@EU',fxDict['USD'],125000,1],
    'DX':['@DX',fxDict['USD'],1000,-1],
    'EBL':['BD',fxDict['EUR'],1000,-1],
    'EBM':['BL',fxDict['EUR'],1000,-1],
    'EBS':['EZ',fxDict['EUR'],1000,-1],
    'ED':['@ED',fxDict['USD'],2500,-1],
    'EMD':['@EMD',fxDict['USD'],100,1],
    'ES':['@ES',fxDict['USD'],50,1],
    'FC':['@GF',fxDict['USD'],500,1],
    'FCH':['MT',fxDict['EUR'],10,1],
    'FDX':['DXM',fxDict['EUR'],5,1],
    'FEI':['IE',fxDict['EUR'],2500,-1],
    'FFI':['LF',fxDict['GBP'],10,1],
    'FLG':['LG',fxDict['GBP'],1000,-1],
    'FSS':['LL',fxDict['GBP'],1250,-1],
    'FV':['@FV',fxDict['USD'],1000,-1],
    'GC':['QGC',fxDict['USD'],100,-1],
    'HCM':['HHI',fxDict['HKD'],50,1],
    'HG':['QHG',fxDict['USD'],250,1],
    'HIC':['HSI',fxDict['HKD'],50,1],
    'HO':['QHO',fxDict['USD'],42000,1],
    'JY':['@JY',fxDict['USD'],125000,-1],
    'KC':['@KC',fxDict['USD'],375,1],
    'KW':['@KW',fxDict['USD'],50,1],
    'LB':['@LB',fxDict['USD'],110,1],
    'LC':['@LE',fxDict['USD'],400,1],
    'LCO':['EB',fxDict['USD'],1000,1],
    'LGO':['GAS',fxDict['USD'],100,1],
    'LH':['@HE',fxDict['USD'],400,1],
    'LRC':['LRC',fxDict['USD'],10,1],
    'LSU':['QW',fxDict['USD'],50,1],
    'MEM':['@MME',fxDict['USD'],50,1],
    'MFX':['IB',fxDict['EUR'],10,1],
    'MP':['@PX',fxDict['USD'],500000,1],
    'MW':['@MW',fxDict['USD'],50,1],
    'NE':['@NE',fxDict['USD'],100000,1],
    'NG':['QNG',fxDict['USD'],10000,1],
    'NIY':['@NKD',fxDict['JPY'],500,1],
    'NQ':['@NQ',fxDict['USD'],20,1],
    'O':['@O',fxDict['USD'],50,1],
    'OJ':['@OJ',fxDict['USD'],150,1],
    'PA':['QPA',fxDict['USD'],100,1],
    'PL':['QPL',fxDict['USD'],50,-1],
    'RB':['QRB',fxDict['USD'],42000,1],
    'RR':['@RR',fxDict['USD'],2000,1],
    'RS':['@RS',fxDict['CAD'],20,1],
    'S':['@S',fxDict['USD'],50,1],
    'SB':['@SB',fxDict['USD'],1120,1],
    'SF':['@SF',fxDict['USD'],125000,1],
    'SI':['QSI',fxDict['USD'],50,-1],
    'SIN':['IN',fxDict['USD'],2,1],
    'SJB':['BB',fxDict['JPY'],100000,-1],
    'SM':['@SM',fxDict['USD'],100,1],
    'SMI':['SW',fxDict['CHF'],10,1],
    'SSG':['SS',fxDict['SGD'],200,1],
    'STW':['TW',fxDict['USD'],100,1],
    'SXE':['EX',fxDict['EUR'],10,1],
    'TF':['@TFS',fxDict['USD'],100,1],
    'TU':['@TU',fxDict['USD'],2000,-1],
    'TY':['@TY',fxDict['USD'],1000,-1],
    'US':['@US',fxDict['USD'],1000,-1],
    'VX':['@VX',fxDict['USD'],1000,-1],
    'W':['@W',fxDict['USD'],50,1],
    'YA':['AP',fxDict['AUD'],25,1],
    'YB':['HBS',fxDict['AUD'],2400,-1],
    'YM':['@YM',fxDict['USD'],5,1],
    'YT2':['HTS',fxDict['AUD'],2800,-1],
    'YT3':['HXS',fxDict['AUD'],8000,-1],
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
    if i==0:
        lastDate = data.index[-1]
    else:
        if data.index[-1]> lastDate:
            lastDate=data.index[-1]
    #print pc
    #print sym, atr,data.tail()
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    contractYear=str(data.R[-1])[3]
    contractMonth=str(data.R[-1])[-2:]
    contractName=c2contractSpec[sym][0]+months[int(contractMonth)]+contractYear
    #print sym, atr[-1], c2contractSpec[sym][2], c2contractSpec[sym][1]
    usdATR = atr[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
    cValue = data.Close[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
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
    futuresDF.set_value(sym,'contractValue',cValue)
    futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
    futuresDF.set_value(sym,'RiskOn',c2contractSpec[sym][3])
futuresDF.index.name = lastDate
    

    
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

if savePath2 != None:
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
    filename=version+'_'+col+'_CORREL'+'.png'
    if savePath2 != None:
        print i+1,'Saving '+savePath2+filename
        plt.savefig(savePath2+filename, bbox_inches='tight')
    
    if len(sys.argv)==1 and showPlots:
        #print data.index[0],'to',data.index[-1]
        plt.show()
    plt.close()

for i2,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename=version+'_'+sym+'.csv'

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
   

for i2,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='0.5_'+sym+'_1D.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    data.index = pd.to_datetime(data.index,format='%Y-%m-%d')
    if i2==0:
        sigDate = data.index[-1]
    else:
        if data.index[-1]> sigDate:
            sigDate=data.index[-1]
            
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

for i2,contract in enumerate(marketList):
    #print i,
    if 'YT' not in contract:
        sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        sym=contract
    signalFilename='0.75_'+sym+'_1D.csv'
    #print signalFilename
    data = pd.read_csv(signalPath+signalFilename, index_col=0)
    if sym in offline:
        adjQty=0
    else:
        if data.dpsSafef.iloc[-1] ==1:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
        else:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
            
    futuresDF.set_value(sym,'0.75LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'0.75LastSAFEf',data.dpsSafef.iloc[-1])
    futuresDF.set_value(sym,'0.75finalQTY',adjQty)
    futuresDF.set_value(sym,'0.75SIG'+str(data.index[-1]),data.signals.iloc[-1])

for i2,contract in enumerate(marketList):
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
columns = futuresDF.columns.tolist()
start_idx =columns.index('ACT')+1
nextColOrder = ['0.75LastSIG','0.5LastSIG','1LastSIG','LastSEA','LastSRUN','vSTART']
new_order = columns[:start_idx]+nextColOrder
new_order =  new_order+[x for x in columns if x not in new_order]
futuresDF = futuresDF[new_order]
print futuresDF.iloc[:,:4]
print 'Saving', savePath+'futuresATR.csv'
futuresDF.to_csv(savePath+'futuresATR.csv')


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
print 'Saving', systemPath+systemFilename2
system.to_csv(systemPath+systemFilename2, index=False)
#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')

c2system='0.5LastSIG'
c2safef=1
signals = ['LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','LastSEA','AntiSEA','AdjSEA','Voting','RiskOn','RiskOff']
votingCols = ['0.75LastSIG','0.5LastSIG','1LastSIG','LastSEA','AdjSEA']

if lastDate > sigDate:
    totalsDF = pd.DataFrame()
    #1bi. Run v4size(to update vlookback)
    #calc the previous day's results.
    futuresDF['AntiSEA'] = np.where(futuresDF.LastSEA==1,-1,1)
    futuresDF['AdjSEA'] = np.where(futuresDF.LastSRUN <0, futuresDF.LastSEA*-1, futuresDF.LastSEA)
    futuresDF['Voting']=np.where(futuresDF[votingCols].sum(axis=1)<0,-1,1)
    futuresDF['RiskOff']=np.where(futuresDF.RiskOn<0,1,-1)
    pctChgCol = [x for x in columns if 'PC' in x][0]
    chgValue = futuresDF[pctChgCol]* futuresDF.contractValue
    for sig in signals:
        futuresDF['PNL_'+sig]=chgValue*futuresDF[sig]
    totals =futuresDF[[x for x in futuresDF if 'PNL' in x]].sum()
    for i,value in enumerate(totals):
        totalsDF.set_value(lastDate, totals.index[i], value)
    files = [ f for f in listdir(savePath) if isfile(join(savePath,f)) ]
    filename = 'futuresScenarios.csv'
    if filename not in files:
        print 'Saving', savePath+filename
        totalsDF.to_csv(savePath+filename)
    else:
        pd.read_csv(savePath+filename, index_col=0).append(totalsDF).to_csv(savePath+filename)
        
    filename='futuresATR_'+lastDate.strftime("%Y%m%d%H%M")+'.csv'
    print 'Saving', savePath2+filename
    futuresDF.to_csv(savePath2+filename)
else:

    #1biv. Run v4size (signals and size) and check system.csv for qty,contracts with futuresATR
    #save signals to v4_ signal files for order processing
    nsig=0
    for ticker in futuresDF.index:
        nsig+=1
        signalFile=pd.read_csv(signalSavePath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        #addLine = signalFile.iloc[-1]
        #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        #addLine.name = sst.iloc[-1].name
        addLine = pd.Series(name=lastDate)
        addLine['signals']=futuresDF.ix[ticker][c2system]
        addLine['safef']=c2safef
        addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
        signalFile = signalFile.append(addLine)
        filename=signalSavePath + version+'_'+ ticker+ '.csv'
        print 'Saving...',  filename
        signalFile.to_csv(filename, index=True)
    print nsig, 'files updated'
    
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()