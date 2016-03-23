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
import subprocess
from btapi.get_signal import get_v1signal

def generate_sigplots(counter, html, cols):
    subprocess.call(['python','create_signalPlots.py','1'])
    systemdata=pd.read_csv('./data/systems/system.csv')
    systemdata=systemdata.reset_index()
    systems=dict()
    for i in systemdata.index:
        system=systemdata.ix[i]
        if system['ibsym'] != 'BTC':
          if not systems.has_key(system['System']):
            filename=system['System']
	    if os.path.isfile('./data/results/' + filename + '.png'):
              systems[system['System']]=1
              if system['Version']=='v1':
                  filename = 'v1_' + filename
              (counter, html)=generate_html(filename, counter, html, cols)
    return (counter, html)     
def generate_paper_c2_plot(systemname, initialEquity):
    filename='./data/paper/c2_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet.sort_index(inplace=True)
        dataSet['equitycurve'] = dataSet['balance']
        dataSet['PurePLcurve'] = dataSet['purebalance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity,initialEquity]], columns=['equitycurve','PurePLcurve'])
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
        dataSet['PurePLcurve'] = dataSet['purebalance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity, initialEquity]], columns=['equitycurve','PurePLcurve'])
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
            dataSet['PurePLcurve'] = dataSet['purebalance']
        else:
            dataSet['equitycurve'] = dataSet['balance']
            dataSet['PurePLcurve'] = dataSet['purebalance']
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
            dataSet['PurePLcurve'] = initialEquity + dataSet['PurePL'].cumsum()
        else:
            dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
            dataSet['PurePLcurve'] = initialEquity + dataSet['PurePL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet

def get_data(systemname, api, broker, dataType, initialData):
    filename='./data/' + api + '/' + broker + '_' + systemname + '_' + dataType + '.csv'
    if api == 'c2api' or api=='ibapi' or api=='btapi':
        filename='./data/' + api + '/' + systemname + '_' + dataType + '.csv'
    print filename
    dataSet=pd.DataFrame([[initialData]], columns=[dataType])
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        return dataSet
    else:
        return dataSet

def get_USD(currency):
    data=pd.read_csv('./data/systems/currency.csv')
    conversion=float(data.loc[data['Symbol']==currency].iloc[-1]['Ask'])
    return float(conversion)
    
def get_datas(systems, api, dataType, initialData):
    dataPath='./data/' + api + '/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    dataSet=pd.DataFrame({}, columns=['Date'])
    dataSet=dataSet.set_index('Date')
    newfiles=list()
    for symbol in systems:   
        search=re.compile(symbol)      
        for file in files:
            if re.search(search, file):        
                filename=dataPath+file
                if os.path.isfile(filename):
                    print ' Price Feed: ' + filename + ' data '+ dataType
    
                    newfiles.append([filename,symbol])
                    
    return newfiles
                
        
def generate_plot(data, systemname, title, ylabel, counter, html, cols=4):
    
    data.plot()   
    fig = plt.figure(1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig('./data/results/' + systemname + ylabel + '.png')
    plt.close(fig)
    (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    
    return (counter, html)

def generate_mult_plot(datas, systemname, title, ylabel, counter, html, cols=4):
    for data in datas:
        data.plot()   
    fig = plt.figure(1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig('./data/results/' + systemname + ylabel + '.png')
    plt.close(fig)
    (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    
    return (counter, html)
    
def generate_plots(datas, systemname, title, ylabel, counter, html, cols=4):
    for (filename, ticker) in datas:
        dta=pd.read_csv(filename)
        symbol=ticker[0:3]
        currency=ticker[3:6]
        #print 'plot for ticker: ' + currency
        if currency != 'USD':
            dta[ylabel]=dta[ylabel] * get_USD(currency)
        dta[ylabel].tail(2000).plot()   
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
c2dict={}
for i in systemdata.index:
    
    system=systemdata.ix[i]
    print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
    print        " System Algo: " + str(system['System']) 
    if system['c2submit']:
	c2dict[system['Name']]=1  
    if not systemdict.has_key(system['Name']):
        systemdict[system['Name']]=list()
        systemdict[system['Name']].append(system['ibsym']+ system['ibcur'])
    else:
        systemdict[system['Name']].append(system['ibsym']+ system['ibcur'])

#Paper
html='<html><head><meta http-equiv="refresh" content="300"></head><body>'
html = html + '<h1>C2</h1><br><table>'
counter=0
cols=3
#Signals
(counter, html)=generate_sigplots(counter, html, cols)
#C2
for systemname in systemdict:
    if c2dict.has_key(systemname):
        c2data=generate_c2_plot(systemname, 20000)
        (counter, html)=generate_plot(c2data['equitycurve'], 'c2_' + systemname+'Equity', 'c2_' + systemname + ' Equity', 'Equity', counter, html, cols)
        
        data=get_data(systemname, 'c2api', 'c2', 'trades', 20000)
        (counter, html)=generate_plot(data['PL'], 'c2_' + systemname+'PL', 'c2_' + systemname + ' PL', 'PL', counter, html, cols)
        
        data=get_datas(systemdict[systemname], 'from_IB', 'Close', 20000)
        (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols)

html = html + '</table><h1>IB</h1><br><table>'
#IB
cols=3
ibdata=generate_ib_plot('IB_Paper', 20000)
(counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'ib_paper', 'IB Live - Equity', 'Equity', counter, html, cols)

ibdata=generate_ib_plot_from_trades('IB_Paper', 20000)
(counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'ib_paper2', 'IB Live - IB Paper From Trades', 'Equity', counter, html, cols)

data=get_data('IB_Live', 'paper', 'ib', 'trades', 20000)
(counter, html)=generate_mult_plot([data['realized_PnL'],data['PurePL']], 'ib_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)
        
ibdata=generate_ib_plot('C2_Paper', 20000)
(counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'ib_c2', 'IB Live - C2 Paper', 'Equity', counter, html, cols)

ibdata=generate_ib_plot_from_trades('C2_Paper', 20000)
(counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'ib_c2_2', 'IB Live - C2 Paper From Trades', 'Equity', counter, html, cols)

data=get_data('IB_Live', 'paper', 'c2', 'trades', 20000)
(counter, html)=generate_mult_plot([data['PL'],data['PurePL']], 'c2_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)
cols=3
       
html = html + '</table><h1>Paper</h1><br><table>'
counter=0
for systemname in systemdict:

  if systemname != 'stratBTC':
    c2data=generate_paper_c2_plot(systemname, 20000)
    (counter, html)=generate_mult_plot([c2data['equitycurve'],c2data['PurePLcurve']], 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html, cols)

    data=get_data(systemname, 'paper', 'c2', 'trades', 20000)
    (counter, html)=generate_mult_plot([data['PL'],data['PurePL']], 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html, cols)

    data=get_datas(systemdict[systemname], 'from_IB', 'Close', 20000)
    (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols)

    ibdata=generate_paper_ib_plot(systemname, 20000)
    (counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html, cols)

    data=get_data(systemname, 'paper', 'ib', 'trades', 20000)
    (counter, html)=generate_mult_plot([data['realized_PnL'],data['PurePL']], 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html, cols)
    
    data=get_datas(systemdict[systemname], 'from_IB', 'Close', 20000)
    (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols)

html = html + '</table><h1>BTC Paper</h1><br><table>'
counter = 0
cols=4

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
                                c2data=generate_paper_c2_plot(systemname, 20000)                                   
                                (counter, html)=generate_mult_plot([c2data['equitycurve'],c2data['PurePLcurve']], 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html)

                                data=get_data(systemname, 'paper', 'c2', 'trades', 20000)
                                (counter, html)=generate_mult_plot([data['PL'],data['PurePL']], 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html)
                        else:
                                systemname=file
                                systemname = re.sub('ib_','', systemname.rstrip())
                                systemname = re.sub('_trades.csv','', systemname.rstrip())
                                ibdata=generate_paper_ib_plot(systemname, 20000)
                                (counter, html)=generate_mult_plot([ibdata['equitycurve'],ibdata['PurePLcurve']], 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html)

                                data=get_data(systemname, 'paper', 'ib', 'trades', 20000)
                                (counter, html)=generate_mult_plot([data['realized_PnL'],data['PurePL']], 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html)
                        btcname=re.sub('stratBTC','BTCUSD',systemname.rstrip())
                            
                        #data=get_datas(btcname, 'btapi', 'Close', 20000)
                        #(counter, html)=generate_plots(data, 'paper_' +btcname + 'Close', btcname + " Close Price", 'Close', counter, html, cols)

                        (counter, html)=generate_html('TWR_' + btcname, counter, html, cols)
                        (counter, html)=generate_html('OHLC_paper_' + btcname+'Close', counter, html, cols)
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
                    generate_plot(data['Close'], 'OHLC_paper_' + systemname, 'OHLC_paper_' + systemname, 'Close', counter, html)
                    plt.close()
                    get_v1signal(data.tail(2000), 'BTCUSD_' + systemname, systemname, True, True, './data/results/TWR_' + systemname + '.png')
                    plt.close()
html = html + '</body></html>'
f = open('./data/results/index.html', 'w')
f.write(html)
f.close()

    #adj_size(model, system['System'],system['Name'],pricefeed,\
    #    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
    #        system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
    #time.sleep(1)



