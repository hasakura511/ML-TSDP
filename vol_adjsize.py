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
riskEquity_mini=500
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
    debug=True
    showPlots=False
    dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    
    #test last>old
    #dataPath2=savePath2
    #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    
    #test last=old
    dataPath2='D:/ML-TSDP/data/'
    
    signalPath ='D:/ML-TSDP/data/signals/'
    signalSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
    
else:
    debug=False
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
        
offline = ['AC','CGB','EBS','ED','FEI','FSS','LB','YB']
offline_mini = ['AC','AEX','CGB','DX','EBL','EBM','EBS','ED','EMD','ES','FC','FCH','FDX','FEI','FFI','FLG','FSS','HCM','HIC','HO','KC','KW','LB','LC','LCO','LH','LRC','LSU','MFX','MP','MW','NE','NIY','O','OJ','S','SF','SI','SM','SMI','SSG','SXE','TF','TU','TY','VX','YA','YB','YM','YT2']


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
    'AC':['@AC',fxDict['USD'],29000,1,'energy'],
    'AD':['@AD',fxDict['USD'],100000,1,'currency'],
    'AEX':['AEX',fxDict['EUR'],200,1,'index'],
    'BO':['@BO',fxDict['USD'],600,1,'grain'],
    'BP':['@BP',fxDict['USD'],62500,1,'currency'],
    'C':['@C',fxDict['USD'],50,1,'grain'],
    'CC':['@CC',fxDict['USD'],10,1,'soft'],
    'CD':['@CD',fxDict['USD'],100000,1,'currency'],
    'CGB':['CB',fxDict['CAD'],1000,-1,'rates'],
    'CL':['QCL',fxDict['USD'],1000,1,'energy'],
    'CT':['@CT',fxDict['USD'],500,1,'soft'],
    'CU':['@EU',fxDict['USD'],125000,1,'currency'],
    'DX':['@DX',fxDict['USD'],1000,-1,'currency'],
    'EBL':['BD',fxDict['EUR'],1000,-1,'rates'],
    'EBM':['BL',fxDict['EUR'],1000,-1,'rates'],
    'EBS':['EZ',fxDict['EUR'],1000,-1,'rates'],
    'ED':['@ED',fxDict['USD'],2500,-1,'rates'],
    'EMD':['@EMD',fxDict['USD'],100,1,'index'],
    'ES':['@ES',fxDict['USD'],50,1,'index'],
    'FC':['@GF',fxDict['USD'],500,1,'meat'],
    'FCH':['MT',fxDict['EUR'],10,1,'index'],
    'FDX':['DXM',fxDict['EUR'],5,1,'index'],
    'FEI':['IE',fxDict['EUR'],2500,-1,'rates'],
    'FFI':['LF',fxDict['GBP'],10,1,'index'],
    'FLG':['LG',fxDict['GBP'],1000,-1,'rates'],
    'FSS':['LL',fxDict['GBP'],1250,-1,'rates'],
    'FV':['@FV',fxDict['USD'],1000,-1,'rates'],
    'GC':['QGC',fxDict['USD'],100,-1,'metal'],
    'HCM':['HHI',fxDict['HKD'],50,1,'index'],
    'HG':['QHG',fxDict['USD'],250,1,'metal'],
    'HIC':['HSI',fxDict['HKD'],50,1,'index'],
    'HO':['QHO',fxDict['USD'],42000,1,'energy'],
    'JY':['@JY',fxDict['USD'],125000,-1,'currency'],
    'KC':['@KC',fxDict['USD'],375,1,'soft'],
    'KW':['@KW',fxDict['USD'],50,1,'grain'],
    'LB':['@LB',fxDict['USD'],110,1,'soft'],
    'LC':['@LE',fxDict['USD'],400,1,'meat'],
    'LCO':['EB',fxDict['USD'],1000,1,'energy'],
    'LGO':['GAS',fxDict['USD'],100,1,'energy'],
    'LH':['@HE',fxDict['USD'],400,1,'meat'],
    'LRC':['LRC',fxDict['USD'],10,1,'soft'],
    'LSU':['QW',fxDict['USD'],50,1,'soft'],
    'MEM':['@MME',fxDict['USD'],50,1,'index'],
    'MFX':['IB',fxDict['EUR'],10,1,'index'],
    'MP':['@PX',fxDict['USD'],500000,1,'currency'],
    'MW':['@MW',fxDict['USD'],50,1,'grain'],
    'NE':['@NE',fxDict['USD'],100000,1,'currency'],
    'NG':['QNG',fxDict['USD'],10000,1,'energy'],
    'NIY':['@NKD',fxDict['JPY'],500,1,'index'],
    'NQ':['@NQ',fxDict['USD'],20,1,'index'],
    'O':['@O',fxDict['USD'],50,1,'grain'],
    'OJ':['@OJ',fxDict['USD'],150,1,'soft'],
    'PA':['QPA',fxDict['USD'],100,1,'metal'],
    'PL':['QPL',fxDict['USD'],50,-1,'metal'],
    'RB':['QRB',fxDict['USD'],42000,1,'energy'],
    'RR':['@RR',fxDict['USD'],2000,1,'grain'],
    'RS':['@RS',fxDict['CAD'],20,1,'grain'],
    'S':['@S',fxDict['USD'],50,1,'grain'],
    'SB':['@SB',fxDict['USD'],1120,1,'soft'],
    'SF':['@SF',fxDict['USD'],125000,1,'currency'],
    'SI':['QSI',fxDict['USD'],50,-1,'metal'],
    'SIN':['IN',fxDict['USD'],2,1,'index'],
    'SJB':['BB',fxDict['JPY'],100000,-1,'rates'],
    'SM':['@SM',fxDict['USD'],100,1,'grain'],
    'SMI':['SW',fxDict['CHF'],10,1,'index'],
    'SSG':['SS',fxDict['SGD'],200,1,'index'],
    'STW':['TW',fxDict['USD'],100,1,'index'],
    'SXE':['EX',fxDict['EUR'],10,1,'index'],
    'TF':['@TFS',fxDict['USD'],100,1,'index'],
    'TU':['@TU',fxDict['USD'],2000,-1,'rates'],
    'TY':['@TY',fxDict['USD'],1000,-1,'rates'],
    'US':['@US',fxDict['USD'],1000,-1,'rates'],
    'VX':['@VX',fxDict['USD'],1000,-1,'index'],
    'W':['@W',fxDict['USD'],50,1,'grain'],
    'YA':['AP',fxDict['AUD'],25,1,'index'],
    'YB':['HBS',fxDict['AUD'],2400,-1,'rates'],
    'YM':['@YM',fxDict['USD'],5,1,'index'],
    'YT2':['HTS',fxDict['AUD'],2800,-1,'rates'],
    'YT3':['HXS',fxDict['AUD'],8000,-1,'rates'],
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
futuresDF_old=pd.read_csv(dataPath2+'futuresATR.csv', index_col=0)
#futuresDF_old=pd.read_csv(dataPath2+'futuresATR_Signals.csv', index_col=0)

oldDate=dt.strptime(futuresDF_old.index.name,"%Y-%m-%d %H:%M:%S")
futuresDF=pd.DataFrame()
corrDF=pd.DataFrame()

for i,contract in enumerate(marketList):
    data = pd.read_csv(dataPath+contract+'_B.csv', index_col=0, header=None)[-lookback-1:]
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
    data.index.name = 'Dates'
    atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
    pc=data.Close.pct_change()
    act=np.where(pc<0,-1,1)
    
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
    qty_mini = int(math.ceil(riskEquity_mini/usdATR))
    #print sym, data.R[-1], contractName
    #signalFilename='v4_'+sym+'.csv'
    corrDF[sym]=pc
    futuresDF.set_value(sym,'Contract',contractName)
    futuresDF.set_value(sym,'LastClose',data.Close[-1])
    futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
    futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
    futuresDF.set_value(sym,'ACT',act[-1])
    #futuresDF.set_value(sym,'prevACT',act[-2])
    futuresDF.set_value(sym,'usdATR',usdATR)
    futuresDF.set_value(sym,'QTY',qty)
    futuresDF.set_value(sym,'QTY_MINI',qty_mini)
    futuresDF.set_value(sym,'contractValue',cValue)
    futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
    futuresDF.set_value(sym,'RiskOn',c2contractSpec[sym][3])
    futuresDF.set_value(sym,'group',c2contractSpec[sym][4])
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

#save last seasonal signal for pnl processing
#update correl charts
if lastDate >oldDate:
    futuresDF['prevACT']=futuresDF_old['prevACT']
    futuresDF['prevSEA']=futuresDF_old.LastSEA
    futuresDF['prevSRUN']=futuresDF_old.LastSRUN
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
        if data.safef.iloc[-1] ==1:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
        else:
            adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
        
    futuresDF.set_value(sym,'LastSIG',data.signals.iloc[-1])
    futuresDF.set_value(sym,'LastSAFEf',data.safef.iloc[-1])
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



#system file update
system = pd.read_csv(systemPath+systemFilename)
system_mini = pd.read_csv(systemPath+systemFilename2)

for sys in system.System:
    sym=sys.split('_')[1]
    idx=system[system.System==sys].index[0]
    print sys, sym, system.ix[idx].c2qty,
    system.set_value(idx,'c2qty',int(futuresDF.ix[sym]['finalQTY']))
    print system.ix[idx].c2qty, system.ix[idx].c2sym,
    system.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
    print system.ix[idx].c2sym

for sys in system_mini.System:
    sym=sys.split('_')[1]
    idx=system_mini[system_mini.System==sys].index[0]
    print 'MINI', sys, sym, system_mini.ix[idx].c2qty,
    if sym in offline_mini:
        system_mini.set_value(idx,'c2qty',0)
    else:
        system_mini.set_value(idx,'c2qty',int(futuresDF.ix[sym]['QTY_MINI']))
    print system_mini.ix[idx].c2qty, system_mini.ix[idx].c2sym,
    system_mini.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
    print system_mini.ix[idx].c2sym
    
print 'Saving', systemPath+systemFilename
system.to_csv(systemPath+systemFilename, index=False)
print 'Saving', systemPath+systemFilename2
system_mini.to_csv(systemPath+systemFilename2, index=False)
#signalDF=signalDF.sort_index()
#print signalDF
#signalDF.to_csv(savePath+'futuresSignals.csv')

#use LastSEA for seasonality in c2
c2system='0.5LastSIG'
c2safef=1
signals = ['ACT','prevACT','AntiPrevACT','RiskOn','RiskOff',\
                'LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','Anti1LastSIG','Anti0.75LastSIG',\
                'prevSEA','AntiSEA','AdjSEA','AntiAdjSEA',\
                'Voting','Voting2','Voting3','Voting4','Voting5','Voting6']


if lastDate > sigDate:
    votingCols = ['1LastSIG','prevACT','AntiSEA']
    voting2Cols = ['0.5LastSIG','AntiPrevACT','AdjSEA']
    voting3Cols = ['0.75LastSIG','AntiPrevACT','AntiSEA']
    voting4Cols=['Voting','Voting2','Voting3']
    voting5Cols=['prevACT','Anti1LastSIG','AntiAdjSEA']
    voting6Cols = ['0.5LastSIG','AntiPrevACT','AntiSEA']
    #voting4Cols= votingCols+voting2Cols+voting3Cols
    #1bi. Run v4size(to update vlookback)
    #calc the previous day's results.
    nrows=futuresDF.shape[0]
    totalsDF = pd.DataFrame()
    futuresDF['Anti1LastSIG'] = np.where(futuresDF['1LastSIG']==1,-1,1)
    futuresDF['Anti0.75LastSIG'] = np.where(futuresDF['0.75LastSIG']==1,-1,1)
    futuresDF['AntiSEA'] = np.where(futuresDF.prevSEA==1,-1,1)
    futuresDF['AntiPrevACT'] = np.where(futuresDF.prevACT==1,-1,1)
    futuresDF['AdjSEA'] = np.where(futuresDF.prevSRUN <0, futuresDF.prevSEA*-1, futuresDF.prevSEA)
    futuresDF['AntiAdjSEA'] = np.where(futuresDF.AdjSEA==1,-1,1)
    futuresDF['Voting']=np.where(futuresDF[votingCols].sum(axis=1)<0,-1,1)
    futuresDF['Voting2']=np.where(futuresDF[voting2Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting3']=np.where(futuresDF[voting3Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting4']=np.where(futuresDF[voting4Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting5']=np.where(futuresDF[voting5Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting6']=np.where(futuresDF[voting6Cols].sum(axis=1)<0,-1,1)
    futuresDF['RiskOff']=np.where(futuresDF.RiskOn<0,1,-1)
    pctChgCol = [x for x in columns if 'PC' in x][0]
    futuresDF['chgValue'] = futuresDF[pctChgCol]* futuresDF.contractValue*futuresDF.finalQTY
    cv_online = futuresDF['chgValue'].drop(offline,axis=0)
    for sig in signals:
        futuresDF['PNL_'+sig]=futuresDF['chgValue']*futuresDF[sig]
        totalsDF.set_value(lastDate, 'ACC_'+sig, sum(futuresDF[sig]==futuresDF.ACT)/float(nrows))
        totalsDF.set_value(lastDate, 'L%_'+sig, sum(futuresDF[sig]==1)/float(nrows))
    totals =futuresDF[[x for x in futuresDF if 'PNL' in x]].sum()
    for i,value in enumerate(totals):
        totalsDF.set_value(lastDate, totals.index[i], value)
        
    bygroup = pd.concat([abs(futuresDF['chgValue']), futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
    volByGroupByContract = bygroup.sum()/bygroup.count()
    bygroup2 = pd.concat([futuresDF['chgValue'], futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
    chgByGroupByContract = bygroup2.sum()/bygroup2.count()
    bygroup3 = pd.concat([futuresDF['ACT']==1, futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
    longPerByGroup = bygroup3.sum()/bygroup3.count()
    
    totalsDF.set_value(lastDate, 'Vol_All', abs(cv_online).sum()/cv_online.count())
    for i,value in enumerate(volByGroupByContract['chgValue']):
        totalsDF.set_value(lastDate, 'Vol_'+volByGroupByContract.index[i], value)
    
    totalsDF.set_value(lastDate, 'Chg_All', cv_online.sum()/cv_online.count())
    for i,value in enumerate(chgByGroupByContract['chgValue']):
        totalsDF.set_value(lastDate, 'Chg_'+chgByGroupByContract.index[i], value)
    
    totalsDF.set_value(lastDate, 'L%_All', sum(futuresDF.ACT.drop(offline, axis=0)==1)/float(cv_online.count()))
    for i,value in enumerate(longPerByGroup['ACT']):
        totalsDF.set_value(lastDate, 'L%_'+longPerByGroup.index[i], value)
    
    print totalsDF.sort_index().transpose()
    
    filename='futuresResults_'+lastDate.strftime("%Y%m%d%H%M")+'.csv'
    print 'Saving', savePath2+filename
    totalsDF.sort_index().transpose().to_csv(savePath2+filename)
    
    filename='futuresResults_Last.csv'
    print 'Saving', savePath+filename
    totalsDF.sort_index().transpose().to_csv(savePath+filename)
    
    files = [ f for f in listdir(savePath) if isfile(join(savePath,f)) ]
    filename = 'futuresResultsHistory.csv'
    if filename not in files:
        print 'Saving', savePath+filename
        totalsDF.to_csv(savePath+filename)
    else:
        print 'Saving', savePath+filename
        pd.read_csv(savePath+filename, index_col=0).append(totalsDF).to_csv(savePath+filename)
        
    filename='futuresATR_'+lastDate.strftime("%Y%m%d%H%M")+'.csv'
    print 'Saving', savePath2+filename
    futuresDF.to_csv(savePath2+filename)
    print 'Saving', savePath+'futuresATR_Results.csv'
    futuresDF.to_csv(savePath+'futuresATR_Results.csv')
    
    filename='futuresL_History.csv'
    cols=['L%_currency',
             'L%_energy',
             'L%_grain',
             'L%_index',
             'L%_meat',
             'L%_metal',
             'L%_ACT',
             'L%_rates',
             'L%_soft']
    if filename not in files:
        print 'Saving', savePath+filename
        totalsDF[cols].to_csv(savePath+filename)
    else:
        print 'Saving', savePath+filename
        pd.read_csv(savePath+filename, index_col=0).append(totalsDF[cols]).to_csv(savePath+filename)
        
else:
    votingCols =['1LastSIG','prevACT','AntiSEA']
    voting2Cols = ['0.5LastSIG','AntiPrevACT','AdjSEA']
    voting3Cols = ['0.75LastSIG','AntiPrevACT','AntiSEA']
    voting4Cols=['Voting','Voting2','Voting3']
    voting5Cols=['prevACT','Anti1LastSIG','AntiAdjSEA']
    voting6Cols = ['0.5LastSIG','AntiPrevACT','AntiSEA']
    #voting4Cols= votingCols+voting2Cols+voting3Cols
    futuresDF['Anti1LastSIG'] = np.where(futuresDF['1LastSIG']==1,-1,1)
    futuresDF['Anti0.75LastSIG'] = np.where(futuresDF['0.75LastSIG']==1,-1,1)
    futuresDF['AntiSEA'] = np.where(futuresDF.LastSEA==1,-1,1)
    futuresDF['prevACT'] = futuresDF.ACT
    futuresDF['AntiPrevACT'] = np.where(futuresDF.ACT==1,-1,1)
    futuresDF['AdjSEA'] = np.where(futuresDF.LastSRUN <0, futuresDF.LastSEA*-1, futuresDF.LastSEA)
    futuresDF['AntiAdjSEA'] = np.where(futuresDF.AdjSEA==1,-1,1)
    futuresDF['Voting']=np.where(futuresDF[votingCols].sum(axis=1)<0,-1,1)
    futuresDF['Voting2']=np.where(futuresDF[voting2Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting3']=np.where(futuresDF[voting3Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting4']=np.where(futuresDF[voting4Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting5']=np.where(futuresDF[voting5Cols].sum(axis=1)<0,-1,1)
    futuresDF['Voting6']=np.where(futuresDF[voting6Cols].sum(axis=1)<0,-1,1)
    futuresDF['RiskOff']=np.where(futuresDF.RiskOn<0,1,-1)
    print 'Saving signals from', c2system
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
        print 'Saving...',  addLine['signals'], addLine['safef'], filename
        signalFile.to_csv(filename, index=True)
    print nsig, 'files updated'
        
    print 'Saving', savePath+'futuresATR_Signals.csv'
    futuresDF.to_csv(savePath+'futuresATR_Signals.csv')
    
print 'Saving', savePath+'futuresATR.csv'
futuresDF.to_csv(savePath+'futuresATR.csv')

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()