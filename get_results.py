from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.ticker as tick
import matplotlib.dates as mdates
from os import listdir
from os.path import isfile, join
import re
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, get_ask as get_ib_ask, get_bid as get_ib_bid
from c2api.place_order import place_order as place_c2order
import threading
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
from datetime import datetime as dt, timedelta
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

initCap=1

def generate_sigplots(counter, html, cols):
    global vdict
    vd=vdict.keys()
    vd.sort()
    filename='Versions'
    fn='./data/results/signal_' + filename + '.html'     
    headerhtml=get_html_header()
    headerhtml = headerhtml + '<h1>Signal - ' + filename + '</h1><br><table>'
    headerhtml = re.sub('Index', filename, headerhtml.rstrip())
    body=''
    for ver in vd:
        (counter, body)=generate_html(ver, counter, body, cols) 
    footerhtml=get_html_footer()
    footerhtml = '</table>' + footerhtml
    write_html(fn, headerhtml, footerhtml, body)

    syms=symdict.keys()
    syms.sort()
    for sym in syms:
        filename=sym
        fn='./data/results/signal_' + filename + '.html'
        #html = html + '<li><a href="' + 'signal_' + filename + '.html">'
        #html = html + filename + '</a></li>'
                
        headerhtml=get_html_header()                
        headerhtml = re.sub('Index', filename, headerhtml.rstrip())
        headerhtml = headerhtml 
        counter=0
        body=' '
        files=symdict[sym]
        files.sort()
         
        for file in files:
            counter=0
            body = body + '<h1>Signal - ' + file + '</h1><br><table>'
            for ver in vd:
                v=ver.split('.')[0]
                v2=file.split('_')[0]
                if v == v2:
                    (counter, body)=generate_html(ver, counter, body, cols)
                
            (counter, body)=generate_sig_html(file, counter, body, cols, True)
            body = body + '</table>'
            (body, counter, cols)=gen_paper(body, counter, cols, 2, file)
            
        footerhtml=get_html_footer()
        write_html(fn, headerhtml, footerhtml, body)
                
              
    return (counter, html)     
    
def generate_paper_c2_plot(systemname, dateCol, initialEquity):
    filename='./data/paper/c2_' + systemname + '_account.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        dataSet['equitycurve'] = dataSet['balance'] 
        dataSet['PurePLcurve'] = dataSet['purebalance'] 
        #dataSet['mark_to_mkt'] = dataSet['mark_to_mkt'] - 20000
        #dataSet['pure_mark_to_mkt'] = dataSet['pure_mark_to_mkt'] - 20000
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
        #dataSet['mark_to_mkt'] = dataSet['mark_to_mkt'] 
        #dataSet['pure_mark_to_mkt'] = dataSet['pure_mark_to_mkt'] 
                
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
            dataSet['mark_to_mkt'] = dataSet['mark_to_mkt'] 
            dataSet['pure_mark_to_mkt'] = dataSet['pure_mark_to_mkt'] 
        else:
            dataSet['equitycurve'] = dataSet['balance']
            dataSet['PurePLcurve'] = dataSet['purebalance'] 
            dataSet['mark_to_mkt'] = dataSet['mark_to_mkt'] 
            dataSet['pure_mark_to_mkt'] = dataSet['pure_mark_to_mkt'] 
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
    
def get_datas(systems, api, dataType, initialData, interval=''):
    dataPath='./data/' + api + '/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    
    dataSet=pd.DataFrame({}, columns=['Date'])
    dataSet=dataSet.set_index('Date')
    newfiles=list()
    for symbol in systems:   
        search=re.compile(interval + symbol)      
        for file in files:
            if re.search(search, file):        
                filename=dataPath+file
                if os.path.isfile(filename):
                    print filename + ' data '+ dataType
    
                    newfiles.append([filename,symbol])      
    return newfiles


def generate_paper_TWR(systemname, broker, dateCol, initialEquity):
    filename='./data/paper/' + broker + '_' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #dataSet=dataSet.sort_values(by=[dateCol])
        
        dataSet['equitycurve'] = dataSet['balance'].diff().div(dataSet['balance'].shift(1)) #.pct_change()
        dataSet['PurePLcurve'] = dataSet['purebalance'].diff().div(dataSet['purebalance'].shift(1)) 
        dataSet['mark_to_mkt'] = dataSet['mark_to_mkt'].diff().div(dataSet['mark_to_mkt'].shift(1)) #.pct_change()
        dataSet['pure_mark_to_mkt'] = dataSet['pure_mark_to_mkt'].diff().div(dataSet['pure_mark_to_mkt'].shift(1)) 
        if broker == 'ib':
            dataSet['ls']=dataSet['side']
            dataSet.ix[dataSet['ls']=='SLD','ls']= 'short'
            dataSet.ix[dataSet['ls']=='BOT','ls']= 'long'
            dataSet['Date']=dataSet['times']
            #dataSet=dataSet.set_index('Date')
            #dataSet=dataSet.sort_index()
        else:
            dataSet['ls']=dataSet['long_or_short']
            dataSet['Date']=dataSet['openedWhen']
            #dataSet=dataSet.set_index('Date')
            #dataSet=dataSet.sort_index()
            
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity,initialEquity,'2016-01-01']], columns=['equitycurve','PurePLcurve',dateCol])
        return dataSet
                
def save_plot(colnames, filename, title, ylabel, SST, comments=''):
    SST=SST.fillna(method='pad')
    fig, ax = plt.subplots()
    for col in colnames:
        if SST.shape[0] > 0:
            tdiff=SST.index[-1] - SST.index[0]
            span = round(tdiff.total_seconds()/60)
            tdiff=tdiff.total_seconds()/3600
            if tdiff == 0:
                tdiff=1
            perhour=round(len(SST[col].values)/tdiff,2)
            ax.plot( SST[col], label=str(col) + ' [' + str(SST.shape[0]) + ' records]' + ' ' + str(perhour) + '/hour Total: ' + str(span) + ' mins')
   
    barSize='1 day'
    #if SST.index.to_datetime()[0].time() and not SST.index.to_datetime()[1].time():
    #    barSize = '1 day'
    #else:
    #    barSize = '1 min'
        
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(myFmt)
    legend = ax.legend(loc='upper left', shadow=True)
    
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize(8)
        label.set_fontweight('bold')
        
    # rotate and align the tick labels so they look better

    # use a more precise date string for the x axis locations in the
    # toolbar
    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')

    #minorLocator = MultipleLocator(SST.shape[0])
    #ax.xaxis.set_minor_locator(minorLocator)

    #xticks = ax.xaxis.get_minor_ticks()
    #xticks[1].label1.set_visible(False)

    fig.autofmt_xdate()
    ax.annotate(str(SST.index[-1]), xy=(0.95, -0.02), ha='left', va='top', xycoords='axes fraction', fontsize=10)
    ax.annotate(comments, xy=(0.02, 0.7), ha='left', va='top', xycoords='axes fraction', fontsize=10)
    #plt.axis([0,1.5,0,2.0])
    ax.set_xlim(SST.index[0], SST.index[-1])
    #ax.set_xlabel(str(SST.index[-1]))

    plt.title(title)
    plt.ylabel(ylabel)
    #plt.show()
    plt.savefig(filename)
    plt.close(fig)
    plt.clf()
    plt.cla()
    plt.close() 

def generate_mult_plot(data, colnames, dateCol, systemname, title, ylabel, counter, html, cols=4, recent=-1):
    try:
        logging.info(' ' + systemname + ', ' + title + ', ' + ylabel)
        SST=data.copy()
        SST[dateCol]=pd.to_datetime(SST[dateCol])
        SST=SST.sort_values(by=[dateCol])
        SST=SST.set_index(dateCol)
        comments = str(SST.shape[0]) + ' Record Count\n'
        if ylabel == 'TWR':
            shorts=len(SST.loc[SST['ls']=='short']['ls'])
            longs=len(SST.loc[SST['ls']=='long']['ls'])
            comm=sum(SST['commission'])
            comments = comments + str(shorts) + ' Short Trades\n'
            comments = comments + str(longs) + ' Long Trades\n'
            comments = comments + '$' + str(comm) + ' Total Commission\n'
        if recent > 0: 
                SST=SST.ix[SST.index[-1] - datetime.timedelta(days=recent):]

        filename='./data/results/' + systemname + ylabel + '.png'
        save_plot(colnames, filename, title, ylabel, SST, comments)
        (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
    
    return (counter, html)
    
def generate_plots(datas, systemname, title, ylabel, counter, html, cols=4, recent=-1):
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
               
        if recent > 0:
		SST=SST.ix[SST.index[-1] + pd.DateOffset(-recent):]
        filename='./data/results/' + systemname + ylabel + '.png'
        save_plot(colnames, filename, title, ylabel, SST)
        
        (counter, html)=generate_html( systemname + ylabel, counter, html, cols)
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
    return (counter, html)

def generate_sig_html(signal, counter, html, cols, colspan):
    cols=4
    filename=signal 
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    filename=signal + "_CDF"
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    filename=signal + "_DPS"
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    filename=signal + "_OOS"
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    filename=signal + "_PDF"
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    filename=signal+ "_Params"
    if os.path.isfile('./data/results/' + filename + '.png'):
      (counter, html)=generate_html(filename, counter, html, cols)
    return (counter, html)
    
def generate_html(filename, counter, html, cols, colspan=False):
    height=300
    width=300
    if counter == 0 or colspan:
            html = html + '<tr>'
    html = html + '<td '
    if colspan:
	html=html + 'colspan=' + str(cols)
    html = html + '><center><a href="' + filename + '.png">'
    html = html + '<img src="' + filename + '.png"  width=' + str(width) + ' height=' + str(height) + '></a></center></td>'
    counter = counter + 1
    if counter >= cols or colspan:
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
ibdict={}
verdict={}
vdict={}

iblive=systemdata.ix[systemdata['ibsubmit']==True].reset_index().copy()
print 'IB Live: '  + str(iblive.shape[0])
iblive['Name']='IB_Live'
iblive.index=iblive.index + systemdata.shape[0]
systemdata=systemdata.append(iblive)

for i in systemdata.index:
    
    system=systemdata.ix[i]
    #print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
    #print        " System Algo: " + str(system['System']) 
    if system['c2submit']:
	   c2dict[system['Name']]=1  
    if system['ibsubmit']: 
        ibdict[system['Name']]=1 
    if not systemdict.has_key(system['Name']):
        systemdict[system['Name']]=list()
        systemdict[system['Name']].append(system['ibsym'] + system['ibcur'])
    else:
        systemdict[system['Name']].append(system['ibsym'] + system['ibcur'])
        
    signal=system['System']
    ver=system['Version']
    	
    if re.search(r'v', ver) and re.search(r'v',signal):
    	if ver=='v1.1':
		ver='v1'
    	verdict[system['Name']]=system['Version']
    	vdict[ver]=1
     
    if system['Version'] == 'v1':
        signal='v1_' + signal
        
    if not sigdict.has_key(system['Name']):
        sigdict[system['Name']]=list()
        sigdict[system['Name']].append(signal+'')
    else:
        sigdict[system['Name']].append(signal+'')

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
systems=dict()
systemdata=systemdata.sort_values(by=['c2sym','Version'])
symdict=dict()
for i in systemdata.index:
    system=systemdata.ix[i]
    if system['ibsym'] != 'BTC':
      if not systems.has_key(system['System']):
        filename=system['System']
        if os.path.isfile('./data/results/' + filename + '.png'):
            systems[system['System']]=1
            (ver, sym)=filename.split('_')
            if not symdict.has_key(sym):
                symdict[sym]=list()
            symdict[sym].append(filename)
            
def gen_sig(html, counter, cols):
    counter = 0
    cols=4 #len(vdict.keys())
    (counter, html)=generate_sigplots(counter, html, cols)
    html = html + '</table>'
    return (html, counter, cols)

def gen_c2(html, counter, cols, recent, systemname):
    cols=4
    html = html + '<h1>' + systemname + '</h1><br><table>'
    
    try:
        if c2dict.has_key(systemname):
            (counter, html)=generate_html(verdict[systemname], counter, html, cols, True)
  
            c2data=generate_c2_plot(systemname, 'openedWhen',  initCap)
            (counter, html)=generate_mult_plot(c2data, ['equitycurve'], 'openedWhen', 'c2_' + systemname+'Equity', 'c2_' + systemname + ' Equity', 'Equity', counter, html, cols, recent)
            
            data=get_data(systemname, 'c2api', 'c2', 'trades','openedWhen', initCap)
            (counter, html)=generate_mult_plot(data, ['PL'], 'openedWhen', 'c2_' + systemname+'PL', 'c2_' + systemname + ' PL', 'PL', counter, html, cols, recent)
            
            data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
            (counter, html)=generate_plots(data, 'c2_' + systemname + 'Signals', 'c2_' + systemname + 'Signals', 'equity', counter, html, cols, recent)
    
            data=get_datas(systemdict[systemname], 'from_IB', 'Close', initCap, '1 min_')
            (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols, recent)

    except Exception as e:
            logging.error("get_c2", exc_info=True)
            counter = 0
    html = html + '</table>'
    return (html, counter, cols)
 

def gen_ib(html, counter, cols):
    try:
        
      systemname='IB_Live'
      
      html = html + '<h1>IB Live</h1><center>'
      
      cols=4
      counter=0
      (html, counter, cols)=gen_paper(html, counter, cols, -1, systemname)
      html = html + '</center><h1>Recent Trades</h1><br><center>'
	
      recent=3
      counter=0
      (html, counter,cols)=gen_paper(html, counter, cols, recent, systemname)

    except Exception as e:
        logging.error("gen_ib", exc_info=True)
	counter = 0
    html = html + '</table>'
    return (html, counter, cols)

eqrank=pd.DataFrame({},columns=['System','IB_Bal','IB_PBL','IB_MM','IB_PMM','IB_Start','IB_End','C2_Bal','C2_PBL','C2_MM','C2_PMM','C2_Start','C2_End'])
eqrank.set_index('System')
def gen_eq_rank(systems, recent, html, type='paper'):
    global eqrank
    for systemname in systems:
        print "Ranking " + systemname
        if type == 'paper' or type == 'signal':
            data=generate_paper_c2_plot(systemname, 'Date', initCap)
        elif type == 'c2':
            data=generate_c2_plot(systemname, 'openedWhen', initCap)
            data['Date']=data['openedWhen']
            
        data['Idx']=pd.to_datetime(data['Date'])
        data=data.set_index('Idx').sort_index()    
        if recent > 0: 
                
                data=data.ix[data.index[-1] - datetime.timedelta(days=recent):]  
        c2bal = 0
        if 'equitycurve' in data and data.shape[0] > 0:
            c2bal=data['equitycurve'][-1] - data['equitycurve'][0]
        c2ppnl=0
        if 'PurePLcurve' in data and data.shape[0] > 0:
            c2ppnl=data['PurePLcurve'][-1] - data['PurePLcurve'][0]
        c2mm=0
        if 'mark_to_mkt' in data and data.shape[0] > 0:
            c2mm=data['mark_to_mkt'][-1] - data['mark_to_mkt'][0]
        c2pmm=0
        if 'pure_mark_to_mkt' in data and data.shape[0] > 0:
            c2pmm=data['pure_mark_to_mkt'][-1] - data['pure_mark_to_mkt'][0]
        c2start=data['Date'][0]
        c2end=data['Date'][-1]
        data=generate_paper_ib_plot(systemname, 'Date', initCap)
        data['Idx']=pd.to_datetime(data['Date'])
        data=data.set_index('Idx').sort_index()
        if recent > 0: 
                data=data.ix[data.index[-1] - datetime.timedelta(days=recent):]   
        ibbal=0
        if 'equitycurve' in data and data.shape[0] > 0:
            ibbal=data['equitycurve'][-1] - data['equitycurve'][0]
        ibppnl=0
        if 'PurePLcurve' in data and data.shape[0] > 0:
            ibppnl=data['PurePLcurve'][-1] - data['PurePLcurve'][0]
        ibmm=0
        if 'mark_to_mkt' in data and data.shape[0] > 0:
            ibmm=data['mark_to_mkt'][-1] - data['mark_to_mkt'][0]
        ibpmm=0
        if 'mark_to_mkt' in data and data.shape[0] > 0:
            ibpmm=data['pure_mark_to_mkt'][-1] - data['pure_mark_to_mkt'][0]
        ibstart=data['Date'][0]
        ibend=data['Date'][-1]
        eqrank.ix[systemname]=[systemname, ibbal, ibppnl, ibmm, ibpmm, ibstart, ibend, c2bal, c2ppnl, c2mm, c2pmm, c2start, c2end]
     
    eqrank=eqrank.sort_values(by=['C2_Bal','C2_MM'], ascending=False)    
    html = html + '<center><table>'
    html = html + '<tr><td><h3>System</h3></td>'
    html = html + '<td><h3>C2 Profit</h3></td>'
    html = html + '<td><h3>C2 Pure Profit</h3></td>'
    html = html + '<td><h3>C2 Mark to Market</h3></td>'
    html = html + '<td><h3>C2 Pure Mark to Market</h3></td>'
    html = html + '<td><h3>C2 Start Date</h3></td>'
    html = html + '<td><h3>C2 End Date</h3></td>'
    html = html + '<td><h3>IB Profit</h3></td>'
    html = html + '<td><h3>IB Pure Profit</h3></td>'
    html = html + '<td><h3>IB Mark to Market</h3></td>'
    html = html + '<td><h3>IB Pure Mark to Market</h3></td>'
    html = html + '<td><h3>IB Start Date</h3></td>'
    html = html + '<td><h3>IB End Date</h3></td>'
    html = html + '</tr>'
    for systemname in eqrank.index:
        (system, ibbal, ibppnl, ibmm, ibpmm, ibstart, ibend, c2bal, c2ppnl, c2mm, c2pmm, c2start, c2end)=eqrank.ix[systemname]
        html = html + '<tr><td><li><a href=' + type + '_' + systemname
        if type != 'signal':
            html = html + str(recent) 
        html = html + '.html>' + systemname +'</a></li></td>'
        html = html + '<td>' + str(c2bal) + '</td>'
        html = html + '<td>' + str(c2ppnl) + '</td>'
        html = html + '<td>' + str(c2mm) + '</td>'
        html = html + '<td>' + str(c2pmm) + '</td>'
        html = html + '<td>' + str(c2start) + '</td>'
        html = html + '<td>' + str(c2end) + '</td>'
        html = html + '<td><li><a href=paper' + '_' + systemname
        html = html + str(recent) + '.html>' + str(ibbal) + '</a></td>'
        html = html + '<td>' + str(ibppnl) + '</td>'
        html = html + '<td>' + str(ibmm) + '</td>'
        html = html + '<td>' + str(ibpmm) + '</td>'
        html = html + '<td>' + str(ibstart) + '</td>'
        html = html + '<td>' + str(ibend) + '</td>'
        html = html + '</tr>'
    html = html + '</table></center>'

    eqrank.to_csv('./data/results/' + type + '_eq_recent' + str(recent) +'.csv')
    return (html, eqrank)
#Paper    
def gen_paper(html, counter, cols, recent, systemname):
    html = html + '<h1>Paper - ' + systemname + '</h1><br>'
    counter=0
    cols=4
    
    logging.info ('C2: ' + systemname)
    #C2 Paper
    if os.path.isfile('./data/paper/c2_' + systemname + '_trades.csv'):
            logging.info ('C2:' + systemname)
            try:
                logging.info ('C2:' + systemname)
                html = html + '<center><table>'
                counter=0
                if verdict.has_key(systemname):
                    if os.path.isfile('./data/results/' + systemname + '.png'):
                        (counter, html)=generate_html(systemname, counter, html, cols, False)
                    else:
                        (counter, html)=generate_html(verdict[systemname], counter, html, cols, False)
                twdata=generate_paper_TWR(systemname, 'c2', 'Date', initCap)    
                (counter, html)=generate_mult_plot(twdata,['equitycurve','PurePLcurve','mark_to_mkt','pure_mark_to_mkt'], 'Date', 'paper_' + systemname + 'c2', systemname + " C2 ", 'TWR', counter, html, cols, recent)
            
                html = html + '</table></center><br><center><table>'
                counter=0
                c2data=generate_paper_c2_plot(systemname, 'Date', initCap)
                (counter, html)=generate_mult_plot(c2data,['equitycurve','PurePLcurve','mark_to_mkt','pure_mark_to_mkt'], 'Date', 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html, cols, recent)
            
                data=get_data(systemname, 'paper', 'c2', 'trades', 'openedWhen', initCap)
                (counter, html)=generate_mult_plot(data,['PL','PurePL'], 'openedWhen', 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html, cols, recent)
            
                data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
                (counter, html)=generate_plots(data, 'c2_' + systemname + 'Signals', 'c2_' + systemname + 'Signals', 'equity', counter, html, cols, recent)
            
                data=get_datas(systemdict[systemname], 'from_IB', 'Close', initCap, '1 min_')
                (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols, recent)        
                html = html + '</center></table><br>'
            except Exception as e:
                      logging.error("get_paper", exc_info=True)
                      counter = 0
   
    
    counter=0
    #IB Paper
    if os.path.isfile('./data/paper/ib_' + systemname + '_trades.csv'):
            logging.info ('IB: ' + systemname)
            try:
                  logging.info ('IB: ' + systemname)
                  html = html + '<center><table>'
                  counter=0
                  if verdict.has_key(systemname):
                      if os.path.isfile('./data/results/' + systemname + '.png'):
                        (counter, html)=generate_html(systemname, counter, html, cols, False)
                      else:
                        (counter, html)=generate_html(verdict[systemname], counter, html, cols, False)
                  
                  twdata=generate_paper_TWR(systemname, 'ib', 'Date', initCap)    
                  (counter, html)=generate_mult_plot(twdata,['equitycurve','PurePLcurve','mark_to_mkt','pure_mark_to_mkt'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'TWR', counter, html, cols, recent)
                              
                  html = html + '</table></center><br><center><table>'
                  counter=0
                  ibdata=generate_paper_ib_plot(systemname, 'Date', initCap)
                  (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve','mark_to_mkt','pure_mark_to_mkt'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html, cols, recent)
                
                  data=get_data(systemname, 'paper', 'ib', 'trades', 'times', initCap)
                  (counter, html)=generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html, cols, recent)
                  
                  data=get_datas(sigdict[systemname], 'signalPlots', 'equity', 0)
                  (counter, html)=generate_plots(data, 'ib_' + systemname + 'Signals', 'ib_' + systemname + 'Signals', 'equity', counter, html, cols, recent)
                  
                  data=get_datas(systemdict[systemname], 'from_IB', 'Close', initCap, '1 min_')
                  (counter, html)=generate_plots(data, 'paper_' + systemname + 'Close', systemname + " Close Price", 'Close', counter, html, cols, recent)
                  html = html + '</table></center><br>'
                  counter=0
            except Exception as e:
                      logging.error("get_paper", exc_info=True)
                      counter = 0
      
    return (html, counter, cols)
    
def gen_btc(html, counter, cols):
    html = html + '<h1>BTC Paper</h1><br><table>'
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
                        try:
                            if re.search(c2search, file):
                                    systemname=file
                                    systemname = re.sub('c2_','', systemname.rstrip())
                                    systemname = re.sub('_trades.csv','', systemname.rstrip())
                                    
                                    c2data=generate_paper_c2_plot(systemname, 'Date', initCap)
                                    (counter, html)=generate_mult_plot(c2data,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'c2', systemname + " C2 ", 'Equity', counter, html, cols)
                                
                                    data=get_data(systemname, 'paper', 'c2', 'trades', 'openedWhen', initCap)
                                    (counter, html)=generate_mult_plot(data,['PL','PurePL'], 'openedWhen', 'paper_' + systemname + 'c2' + systemname+'PL', 'paper_' + systemname + 'c2' + systemname + ' PL', 'PL', counter, html, cols)
                            else:
                                    systemname=file
                                    systemname = re.sub('ib_','', systemname.rstrip())
                                    systemname = re.sub('_trades.csv','', systemname.rstrip())
                                    
                                    ibdata=generate_paper_ib_plot(systemname, 'Date', initCap)
                                    (counter, html)=generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'paper_' + systemname + 'ib', systemname + " IB ", 'Equity', counter, html, cols)
                                
                                    data=get_data(systemname, 'paper', 'ib', 'trades', 'times', initCap)
                                    (counter, html)=generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + systemname + 'ib' + systemname+'PL', 'paper_' + systemname + 'ib' + systemname + ' PL', 'PL', counter, html, cols)
                            
                            btcname=re.sub('stratBTC','BTCUSD',systemname.rstrip())
                                
                            (counter, html)=generate_html('TWR_' + btcname, counter, html, cols)
                            (counter, html)=generate_html('OHLC_paper_' + btcname+'Close', counter, html, cols)
                        except Exception as e: 
	  		    counter = 0
                            logging.error("get_btc", exc_info=True)
    dataPath='./data/from_IB/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    btcsearch=re.compile('BTCUSD')
    
    for file in files:
            if re.search(btcsearch, file):
                    systemname=file
                    systemname = re.sub(dataPath + 'BTCUSD_','', systemname.rstrip())
                    systemname = re.sub('.csv','', systemname.rstrip())
                    data = pd.read_csv(dataPath + file, index_col='Date')
                    if data.shape[0] > 2000:
                        try:
                            get_v1signal(data.tail(2000), 'BTCUSD_' + systemname, systemname, True, True, './data/results/TWR_' + systemname + '.png')
                            plt.close()
                            data=data.reset_index()
                            generate_mult_plot(data, ['Close'], 'Date', 'OHLC_paper_' + systemname, 'OHLC_paper_' + systemname, 'Close', counter, html)
                            plt.close()
                        except Exception as e:
                            logging.error("get_btc", exc_info=True)
    html = html + '</table>'
    return (html, counter, cols)      

def gen_file(filetype):
    filename='./data/results/index.html'
    headertitle=filetype
    html=''
    counter=0
    cols=3
    genstrat=''
    if len(sys.argv)>2:
	   genstrat=sys.argv[2]
    
    if filetype == 'index':
        headertitle='Systems'
        filename='./data/results/index.html'
        html = html + '<li><a href=sig.html>Signals</a></li>'
        html = html + '<li><a href=c2.html>C2</a></li>'
        html = html + '<li><a href=c2_2.html>Recent C2</a></li>'
        html = html + '<li><a href=ib.html>IB</a></li>'
        html = html + '<li><a href=paper.html>Paper</a></li>'
        html = html + '<li><a href=paper2.html>Recent Paper</a></li>'
        html = html + '<li><a href=btc.html>BTC</a></li>'
    elif filetype == 'sig':
        counter=0
        cols=5
        recent=2
        filename='./data/results/sig.html'
        headertitle='Signals'
        html = html + '<h1>Signals</h1><br>'
        html = html + '<center><li><a href="' + 'signal_' + 'Versions' + '.html">'
        html = html + 'Versions' + '</a></li></center><br>'
        syslist=list()
        for sym in symdict.keys():
            syslist = syslist + symdict[sym]
        (html, eqdata)=gen_eq_rank(syslist, recent, html, 'signal')
        headerhtml=get_html_header()
        footerhtml=get_html_footer()
        headerhtml = re.sub('Index', headertitle, headerhtml.rstrip())
        write_html(filename, headerhtml, footerhtml, html)
        
        (html, counter, cols)=gen_sig(html, counter, cols)
    elif filetype == 'c2' or filetype == 'c2_2':
        #C2
        counter=0
        cols=4    
        filename='./data/results/c2.html'
        
        html = '<h1>C2</h1><br>'
        recent = -1
        filename='./data/results/c2.html'
        headertitle='C2'
        if filetype == 'c2_2':
            filename='./data/results/c2_2.html'
            html = '<h1>C2 Recent History</h1><br>'
            headertitle='C2 Recent History'
            recent = 1
        syslist=c2dict.keys()
        syslist.sort()
        (html, eqdata)=gen_eq_rank(syslist, recent, html, 'c2')
        headerhtml=get_html_header()
        footerhtml=get_html_footer()
        headerhtml = re.sub('Index', headertitle, headerhtml.rstrip())
        write_html(filename, headerhtml, footerhtml, html)
        for systemname in syslist:
            if c2dict.has_key(systemname):
                logging.info(systemname)
                fn='./data/results/c2_' + systemname + str(recent) + '.html'
                if len(genstrat) == 0 or genstrat == systemname:
                    headerhtml=get_html_header()
                    headerhtml = re.sub('Index', systemname, headerhtml.rstrip())
                    (body, counter, cols)=gen_c2('', counter, cols, recent, systemname) 
                    footerhtml=get_html_footer()
                    
                    write_html(fn, headerhtml, footerhtml, body)
        
    elif filetype == 'ib':
        #IB
        counter=0
        cols=3
        filename='./data/results/ib.html'
        headertitle='IB'
        (html, counter, cols)=gen_ib(html, counter, cols)
    elif filetype == 'paper' or filetype == 'paper2':
        html = html + '<h1>Paper</h1><br>'
        recent = -1
        filename='./data/results/paper.html'
        headertitle='Paper'
        if filetype == 'paper2':
            filename='./data/results/paper2.html'
            recent = 1
            headertitle='Paper Recent History'
        syslist=systemdict.keys()
        syslist.sort()
        (html, eqdata)=gen_eq_rank(syslist, recent, html, 'paper')
        headerhtml=get_html_header()
        footerhtml=get_html_footer()
        headerhtml = re.sub('Index', headertitle, headerhtml.rstrip())
        write_html(filename, headerhtml, footerhtml, html)
        for systemname in syslist:
            if systemname != 'stratBTC':
                counter=0
                cols=4
                logging.info(systemname)
                fn='./data/results/paper_' + systemname + str(recent) + '.html'
                html = html + '<li><a href="' + 'paper_' + systemname + str(recent) + '.html">'
                html = html + systemname + '</a></li>'
                
                if len(genstrat) == 0 or genstrat == systemname:
                    headerhtml=get_html_header()
                    headerhtml = re.sub('Index', systemname, headerhtml.rstrip())
                    (body, counter, cols)=gen_paper('', counter, cols, recent, systemname)
                    footerhtml=get_html_footer()
                    
                    write_html(fn, headerhtml, footerhtml, body)
            
    elif filetype == 'btc':
        counter=0
        cols=4
        filename='./data/results/btc.html'
        headertitle='BTC'
        (html, counter, cols)=gen_btc(html, counter, cols)  
        
    headerhtml=get_html_header()
    footerhtml=get_html_footer()
    headerhtml = re.sub('Index', headertitle, headerhtml.rstrip())

    write_html(filename, headerhtml, footerhtml, html)
    
    if filetype == 'sig':
    	logfile = open('/logs/create_signalPlots.log', 'a')
    	subprocess.call(['python','create_signalPlots.py','1'], stdout = logfile, stderr = logfile)
    	logfile.close()

def get_html_header():
    header=open('./data/results/header.html', 'r')
    headerhtml=header.read()
    header.close
    return headerhtml

def get_html_footer():
    footer=open('./data/results/footer.html', 'r') 
    footerhtml=footer.read() 
    footer.close
    return footerhtml

def write_html(filename, headerhtml, footerhtml, body):
    f = open(filename, 'w')
    f.write(headerhtml)
    f.write(body)
    f.write(footerhtml)
    f.close() 
types=['index','sig','c2','c2_2','ib','paper','paper2','btc']
def start_resgen():
    #Prep
    threads = []
    for ft in types:
	if len(sys.argv)>1:
	   if sys.argv[1]==ft:
		gen_file(ft)
	else:
		runtype=sys.argv[1]
        	gen_file(ft)
        	sig_thread = threading.Thread(target=gen_file, args=[ft])
        	sig_thread.daemon=True
        	threads.append(sig_thread)
    [t.start() for t in threads]
    [t.join() for t in threads]
    threads=[]

start_resgen()

