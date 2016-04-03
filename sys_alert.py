
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, proc_history
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
import re
import threading

logging.basicConfig(filename='/logs/sys_alert.log',level=logging.DEBUG)
   
def start_proc():
    interval='1h'
    minDataPoints = 5000
    exchange='IDEALPRO'
    secType='CASH'
    
    pairs=bars.get_currencies()
    threads = []
    #bars.get_hist_bars(pairs, interval, minDataPoints, exchange, secType)
    #bars.create_bars(pairs, interval)
      
    t1 = threading.Thread(target=check_bar, args=[pairs, interval, minDataPoints, exchange, secType])
    t1.daemon=True
    threads.append(t1)
    
    #t2 = threading.Thread(target=bars.create_bars, args=[pairs, interval])
    #t2.daemon=True
    #threads.append(t2)
    
    [t.start() for t in threads]
    #[t.join() for t in threads]
    bars.update_bars(pairs, interval)

def send_alert(msg):
    print 'Alert'
    
check=dict()

def check_bar(interval, searchstr):
    dataPath = './data/from_IB/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    for file in files:
        
        if re.search(r'' + searchstr,file):
            data=pd.read_csv(dataPath + file, 'Date')
            if interval == '30m':
                t1 = threading.Thread(target=start_check, args=[interval, searchstr])
                t1.daemon=True
                threads.append(t1)
                lastDate=str(data.index[-1])
                if check.has_key(lastDate):
                    if check[lastDate] > 3:
                        send_alert(msg)
                        
                if not lastDate.has_key(symbol):
                        lastDate[symbol]=timestamp
                                               
                if lastDate[symbol] < timestamp:
                    returnData=True

def start_check(interval, searchstr):
    dataPath = './data/from_IB/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    for file in files:
        
        if re.search(r'' + searchstr,file):
            data=pd.read_csv(dataPath + file, 'Date')
            if interval == '30m':
                t1 = threading.Thread(target=check_bar, args=[interval, searchstr])
                t1.daemon=True
                threads.append(t1)          
                
                
    time.sleep(interval)