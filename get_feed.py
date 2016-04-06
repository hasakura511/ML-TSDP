
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, proc_history, reconnect_ib
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
import os
import logging
import threading
import seitoolz.bars as bars
from dateutil.parser import parse

logging.basicConfig(filename='/logs/get_feed.log',level=logging.DEBUG)

debug=False
dataPath = './data/from_IB/'
minDataPoints = 10000
exchange='IDEALPRO'
secType='CASH'
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'

tickerId=random.randint(100,9999)
currencyPairsDict=dict()
prepData=dict()

def get_ibfeed(sym, cur):
	get_feed(sym, cur,'IDEALPRO','CASH')

def prep_bar_feed():
    global tickerId
    global currencyPairsDict
    pairs=bars.get_currencies()
    for pair in pairs:
        logging.info( 'Prepping ' + pair )
        filename=dataPath+barSizeSetting+'_'+pair+'.csv'
        data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
        
        symbol = pair[0:3]
        currency = pair[3:6]
       
        if not currencyPairsDict.has_key(pair):
            tickerId=tickerId+1
            currencyPairsDict[pair] = tickerId
        
        if os.path.isfile(filename):
            data=pd.read_csv(filename, index_col='Date')
            data=proc_history(symbol, currency,   exchange, secType, whatToShow, data, filename, currencyPairsDict[pair])
            prepData[pair]=data
        logging.info( 'Done Prepping ' + pair )
            
def get_bar_feed():
    global tickerId
    global currencyPairsDict
    pairs=bars.get_currencies()
    for pair in pairs:
        logging.info(  'Subscribing to ' + pair  )
        filename=dataPath+barSizeSetting+'_'+pair+'.csv'
        
        symbol = pair[0:3]
        currency = pair[3:6]
       
        get_realtimebar(symbol, currency,   exchange, secType, whatToShow, prepData[pair], filename, currencyPairsDict[pair])
        logging.info( 'Done Subscribing to ' + pair  )
        
def get_hist():
    global tickerId
    global currencyPairsDict
    pairs=bars.get_currencies()
    for pair in pairs:
        logging.info(  'Getting History for ' + pair  )
        filename=dataPath+barSizeSetting+'_'+pair+'.csv'
        
        eastern=timezone('US/Eastern')
        endDateTime=dt.now(get_localzone())
        date=endDateTime.astimezone(eastern)
        date=date.strftime("%Y%m%d %H:%M:%S EST")
        
        symbol = pair[0:3]
        currency = pair[3:6]
        
        get_history(date, symbol, currency, exchange, secType, whatToShow, prepData[pair], filename, currencyPairsDict[pair], minDataPoints, durationStr, barSizeSetting)

        logging.info( 'Done Getting History for ' + pair  )
        
        
def check_bar():
    pairs=bars.get_currencies()
    dataPath = './data/from_IB/'
    barPath='./data/bars/'
    interval = '1 min'
    finished=False
    time.sleep(120)
    while not finished:
        try:
            count=0
            for pair in pairs:
                dataFile=dataPath + interval + '_' + pair + '.csv'
                barFile=barPath + pair + '.csv'
                
                if os.path.isfile(dataFile) and os.path.isfile(barFile):
                    #data=pd.read_csv(dataFile, index_col='Date')
                    bar=pd.read_csv(barFile, index_col='Date')
                    eastern=timezone('US/Eastern')
                    
                    #timestamp
                    #dataDate=parse(data.index[-1]).replace(tzinfo=eastern)
                    barDate=parse(bar.index[-1]).replace(tzinfo=eastern)
                    nowDate=datetime.datetime.now(get_localzone()).astimezone(eastern)
                    #dtimestamp = time.mktime(dataDate.timetuple())
                    btimestamp = time.mktime(barDate.timetuple())
                    timestamp=time.mktime(nowDate.timetuple()) + 3600
                    checktime = 3
                    
                    checktime = checktime * 60
                    logging.error(pair + ' Feed Last Received ' + str(round((timestamp - btimestamp)/60, 2)) + ' mins ago')
                        
                    if timestamp - btimestamp > checktime:
                        logging.error(pair + ' Feed not being received for ' + str(round((timestamp - btimestamp)/60, 2))) + ' mins'
                        count = count + 1

            if count > 5:
                logging.error('Feed not being received - restarting')
                reconnect_ib()
                start_feed()
                time.sleep(120)
            time.sleep(30)
        except Exception as e:
            logging.error("check_bar", exc_info=True)
            
def start_feed():
    prep_bar_feed()
    
    threads = []
    feed_thread = threading.Thread(target=get_bar_feed)
    feed_thread.daemon=True
    threads.append(feed_thread)
    
    hist_thread = threading.Thread(target=get_hist)
    hist_thread.daemon=True
    threads.append(hist_thread)
    
    [t.start() for t in threads]
    [t.join() for t in threads]


start_feed()
check_bar()

