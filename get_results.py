import numpy as np
import pandas as pd
import time
import matplotlib.ticker as tick
import matplotlib.dates as mdates
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
import logging

logging.basicConfig(filename='/logs/get_results.log',level=logging.DEBUG)


def generate_sigplots(counter, html, cols):
    logfile = open('/logs/create_signalPlots.log', 'a')
    subprocess.call(['python','create_signalPlots.py','1'], stdout = logfile, stderr = logfile)
    logfile.close()
    systemdata=pd.read_csv('./data/systems/system.csv')
    systemdata=systemdata.reset_index()
    systems=dict()
    systemdata=systemdata.sort_values(by=['c2sym','Version'])
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
    
def generate_paper_c2_plot(systemname, dateCol, initialEquity):
    filename='./data/paper/c2_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        dataSet['equitycurve'] = dataSet['balance']
        dataSet['PurePLcurve'] = dataSet['purebalance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity,initialEquity,'2016-01-01']], columns=['equitycurve','PurePLcurve',dateCol])
        return dataSet
    
def generate_c2_plot(systemname, dateCol, initialEquity):
    filename='./data/c2api/' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity,'2016-01-01']], columns=['equitycurve',dateCol])
        return dataSet
     

    return dataSet
        
def generate_paper_ib_plot(systemname, dateCol, initialEquity):
    filename='./data/paper/ib_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        dataSet['equitycurve'] = dataSet['balance']
        dataSet['PurePLcurve'] = dataSet['purebalance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity, initialEquity, '2016-01-01']], columns=['equitycurve','PurePLcurve',dateCol])
        return dataSet
    
def generate_ib_plot(systemname, dateCol, initialEquity):
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
        dataSet=dataSet.sort_values(by=[dateCol])
        if systemname == 'C2_Paper':
            dataSet['equitycurve'] = dataSet['balance']
            dataSet['PurePLcurve'] = dataSet['purebalance']
        else:
            dataSet['equitycurve'] = dataSet['balance']
            dataSet['PurePLcurve'] = dataSet['purebalance']
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity, dateCol]], columns=['equitycurve', dateCol])
        return dataSet

def generate_ib_plot_from_trades(systemname, dateCol, initialEquity):
    filename='./data/ibapi/trades.csv'
    if systemname == 'IB':
        filename='./data/ibapi/trades.csv'
    if systemname == 'IB_Paper':
        filename='./data/paper/ib_IB_Live_trades.csv' 
    if systemname == 'C2_Paper':
        filename='./data/paper/c2_IB_Live_trades.csv'
        
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        #sums up results to starting acct capital
        if systemname == 'C2_Paper':
            dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
            dataSet['PurePLcurve'] = initialEquity + dataSet['PurePL'].cumsum()
        else:
            dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
            dataSet['PurePLcurve'] = initialEquity + dataSet['PurePL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity,'2016-01-01']], columns=['equitycurve',dateCol])
        return dataSet

def get_data(systemname, api, broker, dataType, dateCol, initialData):
    filename='./data/' + api + '/' + broker + '_' + systemname + '_' + dataType + '.csv'
    if api == 'c2api' or api=='ibapi' or api=='btapi':
        filename='./data/' + api + '/' + systemname + '_' + dataType + '.csv'
    print filename
    dataSet=pd.DataFrame([[initialData,'2016-01-01']], columns=[dataType,dateCol])
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
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
                    print filename + ' data '+ dataType
    
                    newfiles.append([filename,symbol])      
    return newfiles
                
def save_plot(colnames, filename, title, ylabel, SST):
    fig, ax = plt.subplots()
    for col in colnames:
        ax.plot( SST[col], label=col)      
    barSize='1 day'
    if SST.index.to_datetime()[0].time() and not SST.index.to_datetime()[1].time():
        barSize = '1 day'
    else:
        barSize = '1 min'
        
    if barSize != '1 day':
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d %H:%M")
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
         
    else:
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d")
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
           
    # Now add the legend with some customizations.
    legend = ax.legend(loc='best', shadow=True)
    
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize(8)
        label.set_fontweight('bold')
        
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the
    # toolbar

    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.title(title)
    plt.ylabel(ylabel)
    #plt.show()
    plt.savefig(filename)
    plt.close(fig)
    plt.close()
    
def generate_mult_plot(data, colnames, dateCol, systemname, title, ylabel, counter, html, cols=4):
    try:
        SST=data
        SST[dateCol]=pd.to_datetime(SST[dateCol])
        SST=SST.sort_values(by=[dateCol])
        SST=SST.set_index(dateCol)
        
        filename='./data/results/' + systemname + ylabel + '.png'
        save_plot(colnames, filename, title, ylabel, SST)
        (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
    
    return (counter, html)
    
def generate_plots(datas, systemname, title, ylabel, counter, html, cols=4):
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
                
        filename='./data/results/' + systemname + ylabel + '.png'
        save_plot(colnames, filename, title, ylabel, SST)
        
        (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
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
sigdict={}
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
    signal=system['System']
    if system['Version'] == 'v1':
        signal='v1_' + signal
    if not sigdict.has_key(system['Name']):
        sigdict[system['Name']]=list()
        sigdict[system['Name']].append(signal)
    else:
        sigdict[system['Name']].append(signal)

#Paper
html='<html><head><meta http-equiv="refresh" content="600"></head><body>'

counter=0
cols=3
#Signals
html = html + '<h1>Signals</h1><br><table>'
(counter, html)=generate_sigplots(counter, html, cols)

#C2
counter=0
cols=4
html = html + '</table><h1>C2</h1><br><table>'
for systemname in systemdict:
    if c2dict.has_key(systemname):
        c2data=generate_c2_plot(systemname, 'openedWhen',  20000)
        (counter, html)=generate_mult_plot(c2data, ['equitycurve'], 'openedWhen', 'c2_' + systemname+'Equity', 'c2_' + systemname + ' Equity', 'Equity', counter, html, cols)
        
        data=get_data(systemname, 'c2api', 'c2', 'trades','openedWhen', 20000)
        (counter, html)=generate_mult_plot(data, ['PL'], 'openedWhen', 'c2_' + systemname+'PL', 'c2_' + systemname + ' PL', 'PL', counter, html, cols)
        
        data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
        (counter, html)=generate_plots(data, 'c2_' + systemname + 'Signals', 'c2_' + systemname + 'Signals', 'equity', counter, html, cols)

        data=get_datas(systemdict[systemname], 'from_IB', 'Close', 20000)
        (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols)

html = html + '</table><h1>IB</h1><br><table>'
#IB
cols=3
if os.path.isfile('./data/paper/ib_' + 'IB_Live' + '_trades.csv'):
    ibdata=generate_ib_plot('IB_Paper','Date', 20000)
    (counter, html)=generate_mult_plot(ibdata, ['equitycurve','PurePLcurve'], 'Date', 'ib_paper', 'IB Live - Equity', 'Equity', counter, html, cols)
    
    ibdata=generate_ib_plot_from_trades('IB_Paper','times', 20000)
    (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'times', 'ib_paper2', 'IB Live - IB Paper From Trades', 'Equity', counter, html, cols)
    
    data=get_data('IB_Live', 'paper', 'ib', 'trades', 'times',20000)
    (counter, html)=generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'ib_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)

if os.path.isfile('./data/paper/c2_' + 'IB_Live' + '_trades.csv'):
    ibdata=generate_ib_plot('C2_Paper', 'Date', 20000)
    (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'ib_c2', 'IB Live - C2 Paper', 'Equity', counter, html, cols)
    
    ibdata=generate_ib_plot_from_trades('C2_Paper', 'openedWhen', 20000)
    (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'openedWhen', 'ib_c2_2', 'IB Live - C2 Paper From Trades', 'Equity', counter, html, cols)
    
    data=get_data('IB_Live', 'paper', 'c2', 'trades', 'openedWhen', 20000)
    (counter, html)=generate_mult_plot(data,['PL','PurePL'], 'openedWhen', 'c2_' + 'IB_Live' +'PL', 'ib_' + 'IB_Live' + ' PL', 'PL', counter, html, cols)

cols=4

#Paper    
html = html + '</table><h1>Paper</h1><br><table>'
counter=0
for systemname in systemdict:

  if systemname != 'stratBTC':
    #C2 Paper
    if os.path.isfile('./data/paper/c2_' + systemname + '_trades.csv'):
        c2data=generate_paper_c2_plot(systemname, 'Date', 20000)
        (counter, html)=generate_mult_plot(c2data,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html, cols)
    
        data=get_data(systemname, 'paper', 'c2', 'trades', 'openedWhen', 20000)
        (counter, html)=generate_mult_plot(data,['PL','PurePL'], 'openedWhen', 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html, cols)
    
        data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
        (counter, html)=generate_plots(data, 'c2_' + systemname + 'Signals', 'c2_' + systemname + 'Signals', 'equity', counter, html, cols)
    
        data=get_datas(systemdict[systemname], 'from_IB', 'Close', 20000)
        (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols)

    #IB Paper
    if os.path.isfile('./data/paper/c2_' + systemname + '_trades.csv'):
        ibdata=generate_paper_ib_plot(systemname, 'Date', 20000)
        (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html, cols)
    
        data=get_data(systemname, 'paper', 'ib', 'trades', 'times', 20000)
        (counter, html)=generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html, cols)
        
        data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
        (counter, html)=generate_plots(data, 'ib_' + systemname + 'Signals', 'ib_' + systemname + 'Signals', 'equity', counter, html, cols)
    
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
                                
                                c2data=generate_paper_c2_plot(systemname, 'Date', 20000)
                                (counter, html)=generate_mult_plot(c2data,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html, cols)
                            
                                data=get_data(systemname, 'paper', 'c2', 'trades', 'openedWhen', 20000)
                                (counter, html)=generate_mult_plot(data,['PL','PurePL'], 'openedWhen', 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html, cols)
                        else:
                                systemname=file
                                systemname = re.sub('ib_','', systemname.rstrip())
                                systemname = re.sub('_trades.csv','', systemname.rstrip())
                                
                                ibdata=generate_paper_ib_plot(systemname, 'Date', 20000)
                                (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html, cols)
                            
                                data=get_data(systemname, 'paper', 'ib', 'trades', 'times', 20000)
                                (counter, html)=generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html, cols)
                        
                        btcname=re.sub('stratBTC','BTCUSD',systemname.rstrip())
                            
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
                    get_v1signal(data.tail(2000), 'BTCUSD_' + systemname, systemname, True, True, './data/results/TWR_' + systemname + '.png')
                    plt.close()
                    data=data.reset_index()
                    generate_mult_plot(data, ['Close'], 'Date', 'OHLC_paper_' + systemname, 'OHLC_paper_' + systemname, 'Close', counter, html)
                    plt.close()
html = html + '</body></html>'
f = open('./data/results/index.html', 'w')
f.write(html)
f.close()

    #adj_size(model, system['System'],system['Name'],pricefeed,\
    #    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
    #        system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
    #time.sleep(1)



