import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, get_ask as get_ib_ask, get_bid as get_ib_bid
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

#other
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
from pandas.io.json import json_normalize

from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep

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

    
#data Parameters
#cycles = 2
minDataPoints = 2000
exchange='IDEALPRO'
symbol='AUD'
currency='USD'
secType='CASH'
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'
ticker = symbol + currency

currencyPairs = ['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'EURGBP','GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']

currencyPairsDict = {}
tickerId=1
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

for pair in currencyPairs:
    filename=dataPath+barSizeSetting+'_'+pair+'.csv'
    data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
   
    eastern=timezone('US/Eastern')
    endDateTime=dt.now(get_localzone())
    date=endDateTime.astimezone(eastern)
    date=date.strftime("%Y%m%d %H:%M:%S EST")
    
    symbol = pair[0:3]
    currency = pair[3:6]
    tickerId=tickerId+1
        
    currencyPairsDict[pair] = data 
    get_feed(symbol, currency,'IDEALPRO','CASH', tickerId)
     
finished=False
while not finished:
    
    model_pos=get_model_pos(['EURJPY','EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
    dps_model_pos=get_dps_model_pos(['v2_EURJPY','v2_EURUSD','v2_GBPUSD','v2_USDCHF','v2_USDJPY','v2_AUDUSD','v2_USDCAD'])
    
    #subprocess.call(['python', 'get_ibpos.py'])
    #ib_pos=get_ibpos()
    #ib_pos=get_ibpos_from_csv()
    for i in systemdata.index:
        
        system=systemdata.ix[i]
        model=model_pos
        if system['Version'] == 'v1':
                model=model_pos
        elif system['Version'] == 'v2':
                model=dps_model_pos
        
        filename='./data/paper/' + system['Name'] + '_portfolio.csv'
           
        system_pos=model.loc[system['System']]
        system_c2pos_qty=round(system_pos['action']) * system['c2qty']
        system_ibpos_qty=round(system_pos['action']) * system['ibqty']
        
        
        ask=float(get_ib_ask(str(system['ibsym']),str(system['ibcur'])))
        bid=float(get_ib_bid(str(system['ibsym']),str(system['ibcur'])))
        exchange=system['ibexch']
        secType=system['ibtype']
        
        commissionkey=system['ibsym']+system['ibcur']+system['ibexch']
        commission=commissiondata.loc[commissionkey]
        commission_pct=float(commission['Pct'])
        commission_cash=float(commission['Cash'])
        
        if debug:
            print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
            print        " System Algo: " + str(system['System']) 
            print        " Ask: " + str(ask)
            print        " Bid: " + str(bid)
            print        " Commission Pct: " + str(commission_pct*100) + "% Commission Cash: " + str(commission_cash)
            print        " Signal: " + str(system_ibpos_qty)
        pricefeed=pd.DataFrame([[ask, bid, 10000, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
        if ask > 0 and bid > 0:
            adj_size(model, system['System'],system['Name'],pricefeed,\
            str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
                system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
        #time.sleep(1)
        
    time.sleep(1)
    

