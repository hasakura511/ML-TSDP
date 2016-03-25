import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_ask as get_ib_ask, get_bid as get_ib_bid
from c2api.place_order import place_order as place_c2order
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
import os
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
import adfapi.s105 as astrat
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

debug=False

if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug=True

if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = './debug/scored_metrics_'
    equityStatsSavePath = './debug/'
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
else:
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'

    
def gettrades(sysname):
    filename='./data/paper/ib_' + sysname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        #dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
        return dataSet

def refresh_paper(sysname):
    files=['./data/paper/c2_' + sysname + '_account.csv','./data/paper/c2_' + sysname + '_trades.csv', \
    './data/paper/ib_'+ sysname + '_portfolio.csv','./data/paper/c2_' + sysname + '_portfolio.csv',  \
    './data/paper/ib_' + sysname + '_account.csv','./data/paper/ib_' + sysname + '_trades.csv']
    for i in files:
        filename=i
        if os.path.isfile(filename):
            os.remove(filename)
            print 'Deleting ' + filename

def get_results(systemname, pairs):
    
    ibdata=seigraph.generate_paper_ib_plot(systemname, 'Date', 20000)
    seigraph.generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity')
    
    data=seigraph.get_data(sysname, 'paper', 'ib', 'trades', 'times', 20000)
    seigraph.generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL')
    
    data=seigraph.get_datas(np.array(pairs)[:,1], 'from_IB', 'Close', 20000)
    seigraph.generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close')
    
def get_history(datas, systemname, ylabel):
    try:
        SST=pd.DataFrame()
        
        for (filename, ticker) in datas:
            dta=pd.read_csv(filename)
            symbol=ticker[0:3]
            currency=ticker[3:6]
            #print 'plot for ticker: ' + currency
            if ylabel == 'Close':
                diviser=dta.iloc[0][ylabel]
                dta[ylabel]=dta[ylabel] /diviser
                
            #dta[ylabel].plot(label=ticker)   
            data=pd.DataFrame()
            
            data['Date']=pd.to_datetime(dta[dta.columns[0]])
            
            data[ticker]=dta[ylabel]
            data=data.set_index('Date') 
            if len(SST.index.values) < 2:
                SST=data
            else:
                SST=SST.join(data)
        colnames=list()
        for col in SST.columns:
            if col != 'Date' and col != 0:
                colnames.append(col)
        data=SST
        data=data.reset_index()        
        data['timestamp']= data['Date']
        
        data=data.set_index('Date')
        return data
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
    return SST

pairs=[['./data/from_IB/1 min_EURJPY.csv', 'EURJPY'],
       ['./data/from_IB/1 min_USDJPY.csv', 'USDJPY'],
       ['./data/from_IB/1 min_CADJPY.csv', 'CADJPY'],
       ['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY'],
       ['./data/from_IB/1 min_AUDJPY.csv', 'AUDJPY']]
sysname='ADF'
refresh_paper(sysname)
data=gettrades(sysname)
SST=get_history(pairs, sysname, 'Close')

pos=dict()
totalpos=dict()
asks=dict()
bids=dict()
def proc_backtest(systemname, SST):
    for [file1,sym1] in pairs:
        #print "sym: " + sym1
        for [file2,sym2] in pairs:
            if sym1 != sym2:
                symPair=sym1+sym2
                if not pos.has_key(symPair):
                    pos[symPair]=dict()
                confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
                print "Coint Confidence: " + str(confidence) + "%"
                for i in SST.index:
                        priceHist=SST.ix[i]
                        
                        asks[sym1]=priceHist[sym1]
                        bids[sym1]=priceHist[sym1]
                        asks[sym2]=priceHist[sym2]
                        bids[sym2]=priceHist[sym2]
                        timestamp=time.mktime(priceHist['timestamp'].timetuple())
                        bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                        bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                        signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                        if signals and len(signals) >= 1:
                            for signal in signals:
                                (barSym, barSig, barCmt)=signal
                                
                                if pos[symPair].has_key(barSym):
                                    pos[symPair][barSym]=pos[symPair][barSym] + barSig
                                else:
                                    pos[symPair][barSym]=barSig
                                    
                                if totalpos.has_key(barSym):
                                    totalpos[barSym]=totalpos[barSym] + barSig
                                else:
                                    totalpos[barSym]=barSig
                                
                                model=generate_model_manual(barSym, totalpos[barSym], 1)
                                
                                if totalpos[barSym] == 0:
                                    totalpos.pop(barSym, None)
                                    
                                if pos[symPair][barSym] == 0:
                                    pos[symPair].pop(barSym, None)
                                
                                commissionkey=barSym
                                commission_pct=0.00002
                                commission_cash=2
                                if commissionkey in commissiondata.index:
                                    commission=commissiondata.loc[commissionkey]
                                    commission_pct=float(commission['Pct'])
                                    commission_cash=float(commission['Cash'])
                                    
                                ask=float(asks[barSym])
                                bid=float(bids[barSym])
                                exchange=barSym
                                secType='CASH'
                                sym=barSym[0:3]
                                currency=barSym[3:6]
                                pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                                if ask > 0 and bid > 0:
                                   
                                    date=datetime.datetime.fromtimestamp(
                                        int(timestamp)
                                    ).strftime("%Y%m%d %H:%M:%S EST")
                                    print 'Signal: ' + barSym + '[' + str(barSig) + ']@' + str(ask)
                                    adj_size(model, barSym, systemname, pricefeed,   \
                                        systemname,systemname,100000,barSym, secType, True, \
                                            100000, sym,currency,exchange, secType, True, date)
                #time.sleep(1)
proc_backtest(sysname, SST)

#results
systemname=sysname
get_results(systemname, pairs)



#threads = []
#for pair in pairs:
#	sig_thread = threading.Thread(target=runv2, args=[pair])
#	sig_thread.daemon=True
#	threads.append(sig_thread)
#	sig_thread.start()

