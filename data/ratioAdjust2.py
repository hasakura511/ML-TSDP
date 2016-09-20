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
    data2=data.copy(deep=True)
    nrows = data2.shape[0]
    data2['OP']=np.insert((data2[' OPEN'][1:].values-data2[' OPEN'][:-1].values-data2[' RINFO'][1:].values)/data2[' OPEN'][:-1].values,0,0)
    data2['HP']=np.insert((data2[' HIGH'][1:].values-data2[' HIGH'][:-1].values-data2[' RINFO'][1:].values)/data2[' HIGH'][:-1].values,0,0)
    data2['LP']=np.insert((data2[' LOW'][1:].values-data2[' LOW'][:-1].values-data2[' RINFO'][1:].values)/data2[' LOW'][:-1].values,0,0)
    data2['CP']=np.insert((data2[' CLOSE'][1:].values-data2[' CLOSE'][:-1].values-data2[' RINFO'][1:].values)/data2[' CLOSE'][:-1].values,0,0)

    for i in range(0,data2.shape[0]):
        if i==0:
            data2.set_value(data2.index[i],'RO',data2[' OPEN'].iloc[i])
            data2.set_value(data2.index[i],'RH',data2[' HIGH'].iloc[i])
            data2.set_value(data2.index[i],'RL',data2[' LOW'].iloc[i])
            data2.set_value(data2.index[i],'RC',data2[' CLOSE'].iloc[i])
        else:
            data2.set_value(data2.index[i],'RO',round(data2.iloc[i-1]['RO']*(1+data2.iloc[i]['OP'])))
            data2.set_value(data2.index[i],'RH',round(data2.iloc[i-1]['RH']*(1+data2.iloc[i]['HP'])))
            data2.set_value(data2.index[i],'RL',round(data2.iloc[i-1]['RL']*(1+data2.iloc[i]['LP'])))
            data2.set_value(data2.index[i],'RC',round(data2.iloc[i-1]['RC']*(1+data2.iloc[i]['CP'])))
    #data2.to_csv('C:/users/hidemi/desktop/python/debug.csv')
    return pd.concat([data2[['DATE','RO', 'RH', 'RL', 'RC']],data2[[' VOL', ' OI', ' P', ' R',' RINFO']]],axis=1)

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
            
