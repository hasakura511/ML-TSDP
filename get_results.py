import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
import re
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
import os
from btapi.get_signal import get_v1signal

def generate_paper_c2_plot(systemname, initialEquity):
    filename='./data/paper/c2_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet.sort_index(inplace=True)
        dataSet['equitycurve'] = dataSet['balance']

        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
    
def generate_c2_plot(systemname, initialEquity):
    filename='./data/c2api/' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet.sort_index(inplace=True)
        dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
     

    return dataSet
        
def generate_paper_ib_plot(systemname, initialEquity):
    filename='./data/paper/ib_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet['equitycurve'] = dataSet['balance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
    
def generate_ib_plot(systemname, initialEquity):
    filename='./data/ibapi/trades.csv'
    if systemname == 'IB':
        filename='./data/ibapi/trades.csv'
    if systemname == 'IB_Paper':
        filename='./data/paper/ib_IB_Live_account.csv' 
    if systemname == 'C2_Paper':
        filename='./data/paper/c2_IB_Live_account.csv'
        
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        if systemname == 'C2_Paper':
            dataSet.sort_index(inplace=True)
            dataSet['equitycurve'] = dataSet['balance']
        else:
            dataSet['equitycurve'] = dataSet['balance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet

def generate_ib_plot_from_trades(systemname, initialEquity):
    filename='./data/ibapi/trades.csv'
    if systemname == 'IB':
        filename='./data/ibapi/trades.csv'
    if systemname == 'IB_Paper':
        filename='./data/paper/ib_IB_Live_trades.csv' 
    if systemname == 'C2_Paper':
        filename='./data/paper/c2_IB_Live_trades.csv'
        
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        if systemname == 'C2_Paper':
            dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
        else:
            dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet

def get_data(systemname, api, broker, dataType, initialData):
    filename='./data/' + api + '/' + broker + '_' + systemname + '_' + dataType + '.csv'
    if api == 'c2api' or api=='ibapi':
        filename='./data/' + api + '/' + systemname + '_' + dataType + '.csv'
    dataSet=pd.DataFrame([[initialData]], columns=[dataType])
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        return dataSet
    else:
        return dataSet
        
def generate_plot(data, systemname, title, ylabel, counter, html, cols=4):
    
    data.plot()   
    fig = plt.figure(1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig('./data/results/' + systemname + ylabel + '.png')
    plt.close(fig)
    (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    
    return (counter, html)

def generate_html(filename, counter, html, cols):
    height=300
    width=300
    if counter == 0:
            html = html + '<tr>'
    html = html + '<td><img src="' + filename + '.png"  width=' + str(width) + ' height=' + str(height) + '></td>'
    counter = counter + 1
    if counter == cols:
        html = html + '</tr>'
        counter=0
    return (counter, html)
    
            
systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

systemdict={}
for i in systemdata.index:
    
    system=systemdata.ix[i]
    print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
    print        " System Algo: " + str(system['System']) 
    
    systemdict[system['Name']]=system

#Paper
html='<html><head><meta http-equiv="refresh" content="60"></head><body>'
html = html + '<h1>C2</h1><br><table>'
counter=0
#C2
for systemname in systemdict:
    if systemdict[systemname]['c2submit']:
        c2data=generate_c2_plot(systemname, 10000)
        (counter, html)=generate_plot(c2data['equitycurve'], 'c2_' + systemname+'Equity', 'c2_' + systemname + ' Equity', 'Equity', counter, html)
        data=get_data(systemname, 'c2api', 'c2', 'trades', 0)
        (counter, html)=generate_plot(data['PL'], 'c2_' + systemname+'PL', 'c2_' + systemname + ' PL', 'PL', counter, html)
        
html = html + '</table><h1>IB</h1><br><table>'
#IB
cols=3
ibdata=generate_ib_plot('IB_Paper', 10000)
(counter, html)=generate_plot(ibdata['equitycurve'], 'ib_paper', 'IB Live - Equity', 'Equity', counter, html, cols)

ibdata=generate_ib_plot_from_trades('IB_Paper', 10000)
(counter, html)=generate_plot(ibdata['equitycurve'], 'ib_paper2', 'IB Live - IB Paper From Trades', 'Equity', counter, html, cols)

data=get_data('IB_Live', 'paper', 'ib', 'trades', 0)
(counter, html)=generate_plot(data['realized_PnL'], 'ib_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)
        
ibdata=generate_ib_plot('C2_Paper', 10000)
(counter, html)=generate_plot(ibdata['equitycurve'], 'ib_c2', 'IB Live - C2 Paper', 'Equity', counter, html, cols)

ibdata=generate_ib_plot_from_trades('C2_Paper', 10000)
(counter, html)=generate_plot(ibdata['equitycurve'], 'ib_c2_2', 'IB Live - C2 Paper From Trades', 'Equity', counter, html, cols)

data=get_data('IB_Live', 'paper', 'c2', 'trades', 0)
(counter, html)=generate_plot(data['PL'], 'c2_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)
cols=4
       
html = html + '</table><h1>Paper</h1><br><table>'
counter=0
for systemname in systemdict:

  if systemname != 'stratBTC':
    c2data=generate_paper_c2_plot(systemname, 10000)
    (counter, html)=generate_plot(c2data['equitycurve'], 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html)

    data=get_data(systemname, 'paper', 'c2', 'trades', 0)
    (counter, html)=generate_plot(data['PL'], 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html)

    ibdata=generate_paper_ib_plot(systemname, 10000)
    (counter, html)=generate_plot(ibdata['equitycurve'], 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html)

    data=get_data(systemname, 'paper', 'ib', 'trades', 0)
    (counter, html)=generate_plot(data['realized_PnL'], 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html)

html = html + '</table><h1>BTC Paper</h1><br><table>'
counter = 0
cols=3

dataPath='./data/paper/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('stratBTC')
tradesearch=re.compile('trade')
c2search=re.compile('c2')
for file in files:
        if re.search(btcsearch, file):
                if re.search(tradesearch, file):
                        if re.search(c2search, file):
                                systemname=file
                                systemname = re.sub('c2_','', systemname.rstrip())
                                systemname = re.sub('_trades.csv','', systemname.rstrip())
                                print systemname
                                c2data=generate_paper_c2_plot(systemname, 10000)                                   
                                (counter, html)=generate_plot(c2data['equitycurve'], 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html)

                                data=get_data(systemname, 'paper', 'c2', 'trades', 0)
                                (counter, html)=generate_plot(data['PL'], 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html)

                                
                        else:
                                systemname=file
                                systemname = re.sub('ib_','', systemname.rstrip())
                                systemname = re.sub('_trades.csv','', systemname.rstrip())
                                ibdata=generate_paper_ib_plot(systemname, 10000)
                                (counter, html)=generate_plot(ibdata['equitycurve'], 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html)

                                data=get_data(systemname, 'paper', 'ib', 'trades', 0)
                                (counter, html)=generate_plot(data['realized_PnL'], 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html)
                        btcname=re.sub('stratBTC','BTCUSD',systemname.rstrip())
                        (counter, html)=generate_html('TWR_' + btcname, counter, html, cols)
dataPath='./data/btapi/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('BTCUSD')

for file in files:
        if re.search(btcsearch, file):
                systemname=file
                systemname = re.sub(dataPath + 'BTCUSD_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
                data = pd.read_csv(dataPath + file, index_col='Date')
                if data.shape[0] > 2000:
                    get_v1signal(data.tail(2000), 'BTCUSD_' + systemname, systemname, True, True, './data/results/TWR_' + systemname + '.png')
                    
html = html + '</body></html>'
f = open('./data/results/index.html', 'w')
f.write(html)
f.close()

    #adj_size(model, system['System'],system['Name'],pricefeed,\
    #    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
    #        system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
    #time.sleep(1)



