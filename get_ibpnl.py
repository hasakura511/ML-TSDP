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
import os
from pandas.io.json import json_normalize

from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_pos
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
tickerId=1
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

model_pos=get_model_pos(['EURJPY','EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
dps_model_pos=get_dps_model_pos(['v2_EURJPY','v2_EURUSD','v2_GBPUSD','v2_USDCHF','v2_USDJPY','v2_AUDUSD','v2_USDCAD'])

        #subprocess.call(['python', 'get_ibpos.py'])
        #ib_pos=get_ibpos()
        #ib_pos=get_ibpos_from_csv()
def getibtrades():
    filename='./data/ibapi/trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        #dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
        return dataSet

def refresh_paper_iblive():
    files=['./data/paper/c2_IB_Live_account.csv','./data/paper/c2_IB_Live_trades.csv', \
    './data/paper/ib_IB_Live_portfolio.csv','./data/paper/c2_IB_Live_portfolio.csv',  \
    './data/paper/ib_IB_Live_account.csv','./data/paper/ib_IB_Live_trades.csv']
    for i in files:
        filename=i
        if os.path.isfile(filename):
            os.remove(filename)
            print 'Deleting ' + filename

refresh_paper_iblive()

data=getibtrades()
symdict={}
for i in data.index:
        order=data.ix[i]
        sym=order['symbol'] + order['symbol_currency']
        qty=order['qty']
        if order['side'] == 'SLD':
            qty=-qty
        if sym in symdict:
            symdict[sym]=symdict[sym]+qty
        else:
            symdict[sym]=qty
        model=generate_model_pos(sym, symdict[sym], symdict[sym])
        systemname='IB_Live'
           
        system_pos=qty
        system_ibpos_qty=qty
        
        ask=float(order['price'])
        bid=float(order['price'])
        exchange=order['exchange']
        secType='CASH'
        
        commissionkey=sym + exchange
        commission=commissiondata.loc[commissionkey]
        commission_pct=float(commission['Pct'])
        commission_cash=float(commission['Cash'])
        
        pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
        if ask > 0 and bid > 0:
            adj_size(model, sym,systemname,pricefeed,\
            str('IBLive'),'IBLive',1,sym,secType, True, \
                1,order['symbol'],order['symbol_currency'],exchange, secType,True,order['times'])
        #time.sleep(1)
        
    

