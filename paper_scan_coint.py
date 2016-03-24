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
import adfapi.s103 as s103
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf
import re

def get_history(datas, ylabel):
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
#pairs=[['./data/from_IB/1 min_NZDJPY.csv', 'NZDJPY'],['./data/from_IB/1 min_CADJPY.csv', 'CADJPY']]
pairs=list()

#BTAPI
dataPath='./data/btapi/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('BTCUSD')

for file in files:
        if re.search(btcsearch, file):
                systemname=file
                systemname = re.sub('BTCUSD_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
                #pairs.append([dataPath+file, systemname])
dataPath='./data/from_IB/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('1 min')

for file in files:
        if re.search(btcsearch, file):
                systemname=file
                systemname = re.sub('1 min_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
                pairs.append([dataPath+file, systemname])

def scan_coint():
    global pairs
    result=pd.DataFrame({}, columns=['Date','Symbol1','Symbol2','Confidence','Pv','Hurst'])
    for [file1,sym1] in pairs:
        #print "sym: " + sym1
        for [file2,sym2] in pairs:
            if sym1 != sym2:
                try:
                    print "Sym1: " + sym1 + " Sym2: " + sym2
                    SST=get_history([[file1, sym1],[file2,sym2]], 'Close')
                    (confidence,pv, hurst)=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
                    print "Coint Confidence: " + str(confidence) + "%"
                    rec=pd.DataFrame([[int(time.time()), sym1, sym2, confidence, pv, hurst]], columns=['Date','Symbol1','Symbol2','Confidence','Pv','Hurst']).iloc[-1]
                    result=result.append(rec)
                        
                    result.to_csv('./data/adf/coint.csv',index=False)
                except Exception as e:
                    print "Error getting coint"
    
scan_coint()