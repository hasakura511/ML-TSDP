# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

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
import time
import inspect
import os
import os.path
import sys
import ssl
from copy import deepcopy

def refreshData(contract,data):
    path = os.path.join(dataPath, 'F_'+contract+'.txt')

    # check to see if market data is present. If not (or refresh is true), download data from quantiacs.
    '''
    try:
        with open(path, 'w') as dataFile:
            dataFile.write(data)
        print 'Downloading ' + contract

    except:
        print 'Unable to download ' + contract

    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

    for i,f in enumerate(files):
        if 'F_' in f:
            print 'Ratio Adjusting',f,i
            #if 'F_'+contract+'.txt' in files and (ticker[0:3] in contract or ticker[3:6] in contract):
            data = pd.read_csv(dataPath+f)[-minDataPoints:]
            #data.index.name = 'Dates'
    '''       
    data=ratioAdjust(data[-minDataPoints:])
    data.columns = ['Date','Open','High','Low','Close','Volume','OI','P','R','RINFO']
    data.to_csv(path, index=False)

        
def ratioAdjust(data):
    #data2=data.copy(deep=True)
    nrows = data.shape[0]
    data['OP']=np.insert((data[' OPEN'][1:].values-data[' OPEN'][:-1].values-data[' RINFO'][1:].values)/data[' OPEN'][:-1].values,0,0)
    data['HP']=np.insert((data[' HIGH'][1:].values-data[' HIGH'][:-1].values-data[' RINFO'][1:].values)/data[' HIGH'][:-1].values,0,0)
    data['LP']=np.insert((data[' LOW'][1:].values-data[' LOW'][:-1].values-data[' RINFO'][1:].values)/data[' LOW'][:-1].values,0,0)
    data['CP']=np.insert((data[' CLOSE'][1:].values-data[' CLOSE'][:-1].values-data[' RINFO'][1:].values)/data[' CLOSE'][:-1].values,0,0)

    for i in range(0,data.shape[0]):
        if i==0:
            data.set_value(data.index[i],'RO',data[' OPEN'].iloc[i])
            data.set_value(data.index[i],'RH',data[' HIGH'].iloc[i])
            data.set_value(data.index[i],'RL',data[' LOW'].iloc[i])
            data.set_value(data.index[i],'RC',data[' CLOSE'].iloc[i])
        else:
            data.set_value(data.index[i],'RO',round(data.iloc[i-1]['RO']*(1+data.iloc[i]['OP'])))
            data.set_value(data.index[i],'RH',round(data.iloc[i-1]['RH']*(1+data.iloc[i]['HP'])))
            data.set_value(data.index[i],'RL',round(data.iloc[i-1]['RL']*(1+data.iloc[i]['LP'])))
            data.set_value(data.index[i],'RC',round(data.iloc[i-1]['RC']*(1+data.iloc[i]['CP'])))
    #data.to_csv('C:/users/hidemi/desktop/python/debug.csv')
    return pd.concat([data[['DATE','RO', 'RH', 'RL', 'RC']],data[[' VOL', ' OI', ' P', ' R',' RINFO']]],axis=1)

minDataPoints=500
today=int(sys.argv[1])
refresh=False
dataPath='./tickerData/'
with open('./futures.txt') as f:
    marketList = f.read().splitlines()
    
with open('./refreshFutures.txt') as f:
    refreshFutures = f.read().splitlines()
    

      
#print marketList
#print refreshFutures
nMarkets = len(marketList)
nrefreshFutures = len(refreshFutures)
if nrefreshFutures==0:
    refreshFutures=marketList
    nrefreshFutures = len(refreshFutures)
    
#print marketList
print 'Refreshing\n',refreshFutures
if sys.version[:5] is '2.7.9':
    gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

# set up data director
if not os.path.isdir(dataPath):
    os.mkdir(dataPath)
    #refresh=True
refreshFutures2 = deepcopy(refreshFutures)
for j in range(nrefreshFutures):
    contract=refreshFutures[j]
    print j, refreshFutures[j]
    if sys.version[:5] is '2.7.9':
        data = urllib.urlopen('https://www.quantiacs.com/data/F_' +
                              contract+'.txt',
                              context=gcontext).read()
        data = pd.read_csv(io.StringIO(data.decode('utf-8')))
        newdate = int(data.iloc[-1].DATE)
        olddate = int(pd.read_csv(dataPath+'F_'+contract+'.txt').iloc[-1].Date)
        if newdate > olddate:
            print 'Refreshing Data..', contract, newdate
            refreshData(contract, data)
            refreshFutures2.remove(contract)
        else:
            if olddate == today:
                refreshFutures2.remove(contract)
            else:
                print 'No New Data Found for', contract, 'lastDate', olddate



    else:
        data = urllib.urlopen('https://www.quantiacs.com/data/F_' +
                              contract+'.txt').read()
        data = pd.read_csv(io.StringIO(data.decode('utf-8')))
        newdate = int(data.iloc[-1].DATE)
        olddate = int(pd.read_csv(dataPath+'F_'+contract+'.txt').iloc[-1].Date)
        if newdate > olddate:
            print 'Refreshing Data..', contract, newdate
            refreshData(contract, data)
            refreshFutures2.remove(contract)
        else:
            
            if olddate == today:
                refreshFutures2.remove(contract)
            else:
                print 'No New Data Found for', contract, 'lastDate', olddate

with open('./refreshFutures.txt','w') as f:
    for contract in refreshFutures2:
      f.write("%s\n" % contract)
            
