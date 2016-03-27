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
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, proc_history, get_bar
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

rtbar={}
rtdict={}
rthist={}
rtfile={}
rtreqid={}
lastDate={}
tickerId=1
    
def get_currencies():
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
    return currencyPairs


def get_ibfeed(sym, cur):
	get_feed(sym, cur,'IDEALPRO','CASH')


def compress_min_bar(sym, histData, filename, interval='30m'):
    global pricevalue
    global finished
    global rtbar
    global rtdict
    global rtfile
    global rtreqid
    global tickerId
    
    reqId=0
    pair=sym
    if not rtreqid.has_key(pair):
        tickerId=tickerId+1
        reqId=tickerId
        rtdict[reqId]=pair
        rtfile[reqId]=filename
        rtreqid[pair]=reqId
    else:
        reqId=rtreqid[pair]
        
    rtdict[reqId]=sym
    rtfile[reqId]=filename
    rtreqid[sym]=reqId
    if not rtbar.has_key(reqId):
        rtbar[reqId]=pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
    
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
    if interval == '30m':
        mins=int(datetime.datetime.fromtimestamp(
                    int(timestamp)
                ).strftime('%M'))
        if mins < 30:
            #time
            date=datetime.datetime.fromtimestamp(
                        int(timestamp)
                    ).strftime('%Y%m%d  %H:00:00') 
        else:
             date=datetime.datetime.fromtimestamp(
                        int(timestamp)
                    ).strftime('%Y%m%d  %H:30:00') 
    elif interval == '1h':
        date=datetime.datetime.fromtimestamp(
                int(timestamp)
            ).strftime('%Y%m%d  %H:00:00') 
    #time=time.astimezone(eastern).strftime('%Y-%m-%d %H:%M:00') 
    wap=0
    count=data.shape[0]
    
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
            gotbar.to_csv('./data/bars/' + interval + '_' + sym + '.csv')
        
        print "New Bar:   " + sym + " date:" + str(date) + " open: " + str(open) + " high:"  + str(high) + ' low:' + str(low) + ' close: ' + str(close) + ' volume:' + str(volume) 
        data=data.reset_index().append(pd.DataFrame([[date, open, high, low, close, volume]], columns=['Date','Open','High','Low','Close','Volume'])).set_index('Date')
        
    rtbar[reqId]=data
 
def create_bars(currencyPairs, interval='30m'):
    global tickerId
    
    dataPath='./data/from_IB/'
    for pair in currencyPairs:
        filename=dataPath+interval+'_'+pair+'.csv'
        minFile=dataPath+'1 min'+'_'+pair+'.csv'
        symbol = pair
        if os.path.isfile(minFile):
            data=pd.read_csv(minFile)
            for i in data.index:
                quote=data.ix[i]
                compress_min_bar(symbol, quote, filename, interval)
            
def get_hist_bars(currencyPairs, interval='30m', minDataPoints = 10000, exchange='IDEALPRO', secType='CASH'):
    global rtbar
    global rtdict
    global rtfile
    global rtreqid
    global tickerId
    
    dataPath='./data/from_IB/'
    
    for pair in currencyPairs:
        filename=dataPath+interval+'_'+pair+'.csv'
        symbol = pair[0:3]
        currency = pair[3:6]
        
        durationStr='1 D'
        barSizeSetting='1 min'
        if interval == '30m':
            durationStr='30 D'
            barSizeSetting='30 mins'
        elif interval == '1h':
            durationStr='30 D'
            barSizeSetting='1 hour'
        whatToShow='MIDPOINT'
        
        reqId=0
        date=''
        if not rtreqid.has_key(pair):
            
            tickerId=tickerId+1
            reqId=tickerId
            if os.path.isfile(filename):
                rtbar[reqId]=pd.read_csv(filename, index_col='Date')
                date=str(rtbar[reqId].index[0])
            else:
                rtbar[reqId]=pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
                eastern=timezone('US/Eastern')
                endDateTime=dt.now(get_localzone())
                date=endDateTime.astimezone(eastern)
                date=date.strftime("%Y%m%d %H:%M:%S EST")
            rtdict[reqId]=pair
            rtfile[reqId]=filename
            rtreqid[pair]=reqId
            
            print 'Starting: ' + date
        else:
            reqId=rtreqid[pair]
            
        data = rtbar[reqId]
        histdata = get_history(date, symbol, currency, exchange, secType, whatToShow, data, filename, reqId, minDataPoints, durationStr, barSizeSetting)
        
        if len(histdata.index) > 1:
            data = rtbar[reqId]
            data = data.reset_index().set_index('Date')
            histdata=histdata.reset_index().set_index('Date')
            #data.to_csv('test1')
            #histdata.to_csv('test2')
            
            data = data.combine_first(histdata)
            #data.to_csv('test3')
             
            data  =data.reset_index() 
            histdata = histdata.reset_index()
            
            data=data.sort_values(by='Date')  
            quote=data.iloc[-1]
            #print "Close Bar: " + sym + " date:" + str(quote['Date']) + " open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) + ' wap:' + str(wap) 
            data=data.set_index('Date')
            rtbar[reqId]=data
            data.to_csv(filename)
            
            
            #gotbar=pd.DataFrame([[quote['Date'], quote['Open'], quote['High'], quote['Low'], quote['Close'], quote['Volume'], pair]], columns=['Date','Open','High','Low','Close','Volume','Symbol']).set_index('Date')
            #gotbar.to_csv('./data/bars/' + interval + '_' + pair + '.csv')
            #time.sleep(30)


def update_bars(currencyPairs, interval='30m'):
    global tickerId
    global lastDate
    dataPath='./data/from_IB/'
    while 1:
        for pair in currencyPairs:
            filename=dataPath+interval+'_'+pair+'.csv'
            minFile='./data/bars/'+pair+'.csv'
            symbol = pair
            if os.path.isfile(minFile):
                data=pd.read_csv(minFile)
                 
                eastern=timezone('US/Eastern')
                
                date=data.iloc[-1]['Date']
                date=parse(date).replace(tzinfo=eastern)
                timestamp = time.mktime(date.timetuple())
                
                if not lastDate.has_key(symbol):
                    lastDate[symbol]=timestamp
                                   
                if lastDate[symbol] < timestamp:
                    lastDate[symbol]=timestamp
                    quote=data.iloc[-1]
                    compress_min_bar(symbol, quote, filename, interval) 
        time.sleep(20)