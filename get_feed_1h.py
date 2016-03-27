
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
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
from dateutil.parser import parse
currencyList=dict()
v1sList=dict()
dpsList=dict()

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
for i in systemdata.index:
    system=systemdata.ix[i]
    if system['ibsym'] != 'BTC':
     
      currencyList[system['ibsym']+system['ibcur']]=1
      if system['Version'] == 'v1':
          v1sList[system['System']]=1
      else:
          dpsList[system['System']]=1


currencyPairs = currencyList.keys()

start_time = time.time()


debug=False

if len(sys.argv)==1:
    debug=True

if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = './debug/scored_metrics_'
    equityStatsSavePath = './debug/'
    signalPath = './debug/'
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
minDataPoints = 10000
durationStr='1 D'
barSizeSetting='1 hour'
exchange='IDEALPRO'
symbol='AUD'
currency='USD'
secType='CASH'
whatToShow='MIDPOINT'
ticker = symbol + currency

def get_ibfeed(sym, cur):
	get_feed(sym, cur,'IDEALPRO','CASH')
rtbar={}
rtdict={}
rthist={}
rtfile={}

fask={}
fasksize={}
fbid={}
fbidsize={}
fdict={}

def proc_history(sym, currency, exchange, type, whatToShow, histData, filename, tickerId):
    global pricevalue
    global finished
    global rtbar
    global rtdict
    global rtfile
    reqId=tickerId
    rtdict[reqId]=sym
    rtfile[reqId]=filename
    if not rtbar.has_key(reqId):
        rtbar[reqId]=pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume'])
    
    data=rtbar[reqId]
      
    date=histData['Date']
    open=histData['Open']
    high=histData['High']
    low=histData['Low']
    close=histData['Close']
    volume=histData['Volume']
    
    eastern=timezone('US/Eastern')
    #timestamp
    date=parse(date).replace(tzinfo=eastern)
    timestamp = time.mktime(date.timetuple())
    #time
    date=datetime.datetime.fromtimestamp(
                int(timestamp), eastern
            ).strftime('%Y%m%d  %H:00:00') 
    #time=time.astimezone(eastern).strftime('%Y-%m-%d %H:%M:00') 
    wap=0
    
    if date in data.index:
           
        quote=data.loc[date].copy()
        if high > quote['High']:
            quote['High']=high
        if low < quote['Low']:
            quote['Low']=low
        quote['Close']=close
        quote['Volume']=quote['Volume'] + volume
        if quote['Volume'] < 0:
            quote['Volume'] = 0 
        data.loc[date]=quote
        #print "Update Bar: bar: sym: " + sym + " date:" + str(time) + "open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) + ' wap:' + str(wap) + ' count:' + str(count)
    
    else:
        if len(data.index) > 1:
            data=data.reset_index()                
            data=data.sort_values(by='Date')  
            quote=data.iloc[-1]
            print "Close Bar: " + sym + " date:" + str(quote['Date']) + " open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) + ' wap:' + str(wap) 
            data=data.set_index('Date')
            data.to_csv(filename)
            
            gotbar=pd.DataFrame([[quote['Date'], quote['Open'], quote['High'], quote['Low'], quote['Close'], quote['Volume'], sym]], columns=['Date','Open','High','Low','Close','Volume','Symbol']).set_index('Date')
            gotbar.to_csv('./data/bars/1h_' + sym + '.csv')
        
        print "New Bar:   " + sym + " date:" + str(date) + " open: " + str(open) + " high:"  + str(high) + ' low:' + str(low) + ' close: ' + str(close) + ' volume:' + str(volume) 
        data=data.reset_index().append(pd.DataFrame([[date, open, high, low, close, volume]], columns=['Date','Open','High','Low','Close','Volume'])).set_index('Date')
        
        
    rtbar[reqId]=data
    
currencyPairsDict = {}
tickerId=1
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
for pair in currencyPairs:
    filename=dataPath+'1h'+'_'+pair+'.csv'
    minFile=dataPath+'1 min'+'_'+pair+'.csv'
    
    data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
        
    eastern=timezone('US/Eastern')
    endDateTime=dt.now(get_localzone())
    date=endDateTime.astimezone(eastern)
    date=date.strftime("%Y%m%d %H:%M:%S EST")
    
    symbol = pair
    currency = pair[3:6]
    tickerId=tickerId+1
        
    currencyPairsDict[pair] = data
    
    if os.path.isfile(minFile):
        data=pd.read_csv(minFile)
        for i in data.index:
            quote=data.ix[i]
            proc_history(symbol, currency, exchange, secType, whatToShow, quote, filename, tickerId)
        

finished=False
#while not finished:
#    time.sleep(30)
    

