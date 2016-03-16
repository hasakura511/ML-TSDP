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

def generate_paper_c2_plot(systemname, initialEquity):
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
    
def generate_c2_plot(systemname, initialEquity):
    filename='./data/c2api/' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet['equitycurve'] = initialEquity + dataSet['PL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
     

    return dataSet
        
def generate_paper_ib_plot(systemname, initialEquity):
    filename='./data/paper/ib_' + systemname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
        return dataSet
    else:
        dataSet=pd.DataFrame([[initialEquity]], columns=['equitycurve'])
        return dataSet
    
def generate_ib_plot(systemname, initialEquity):
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


systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

systemdict={}
width=300
height=300
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
        
        c2data['equitycurve'].plot()   
        
        fig = plt.figure(1)
        plt.title(systemname)
        plt.ylabel("Equity")
        plt.savefig('./data/results/c2_' + systemname + '.png')
        plt.close(fig)
        if counter == 0:
            html = html + '<tr>'
        html = html + '<td><img src="c2_' + systemname + '.png"  width=' + str(width) + ' height=' + str(height) + '></td>'
        counter = counter + 1
        if counter == 3:
            html = html + '</tr>'
            counter=0
html = html + '</table><h1>IB</h1><br>'
#IB
ibdata=generate_ib_plot('IB_Paper', 10000)
ibdata['equitycurve'].plot()
fig = plt.figure(1)
plt.title('IB Live - IB Paper')
plt.ylabel("Equity")
plt.savefig('./data/results/ib_paper.png')
plt.close(fig)
html = html + '<img src="ib_paper.png"  width=' + str(width) + ' height=' + str(height) + '><br>'

ibdata=generate_ib_plot('C2_Paper', 10000)
ibdata['equitycurve'].plot()
fig = plt.figure(1)
plt.title('IB Live - C2 Paper')
plt.ylabel("Equity")
plt.savefig('./data/results/ib_c2.png')
plt.close(fig)
html = html + '<img src="ib_c2.png"  width=' + str(width) + ' height=' + str(height) + '><br>'

html = html + '<h1>Paper</h1><br><table>'
counter=0
for systemname in systemdict:

 
    c2data=generate_paper_c2_plot(systemname, 10000)
    c2data['equitycurve'].plot()  
    
    fig = plt.figure(1)
    plt.title(systemname + " C2 ")
    plt.ylabel("Equity")
    plt.savefig('./data/results/paper_c2' + systemname + '.png')
    plt.close(fig)
    if counter == 0:
        html = html + '<tr>'
    
    html = html + '<td><img src="paper_c2' + systemname + '.png" width=' + str(width) + ' height=' + str(height) + '><br></td>'
    
    counter = counter + 1
    if counter == 3:
        html = html + '</tr>'
        counter=0
    
    ibdata=generate_paper_ib_plot(systemname, 10000)
    ibdata['equitycurve'].plot()

    fig = plt.figure(1)
    plt.title(systemname + " IB ")
    plt.ylabel("Equity")
    plt.savefig('./data/results/paper_ib' + systemname + '.png')
    plt.close(fig)
    if counter == 0:
        html = html + '<tr>'
    
    html = html + '<td><img src="paper_ib' + systemname + '.png" width=' + str(width) + ' height=' + str(height) + '><br></td>'
    
    counter = counter + 1
    if counter == 3:
        html = html + '</tr>'
        counter=0

html = html + '</table><h1>BTC Paper</h1><br><table>'
counter = 0

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
                                systemname = re.sub(dataPath + 'c2_','', systemname.rstrip())
                                systemname = re.sub('.csv','', systemname.rstrip())
                                c2data=generate_paper_c2_plot(systemname, 10000)
                                c2data['equitycurve'].plot()  
                                fig = plt.figure(1)
                                
                                
                                plt.title(systemname + " C2 ")
                                    
                                plt.ylabel("Equity")
                                plt.savefig('./data/results/paper_c2' + systemname + '.png')
                                plt.close(fig)
                                
                                html = html + '<td><img src="paper_c2' + systemname + '.png" width=' + str(width) + ' height=' + str(height) + '><br></td>'
                        
                        else:
                                systemname=file
                                systemname = re.sub(dataPath + 'ib_','', systemname.rstrip())
                                systemname = re.sub('.csv','', systemname.rstrip())
                                ibdata=generate_paper_ib_plot(systemname, 10000)
                                ibdata['equitycurve'].plot()
                            
                                fig = plt.figure(1)
                                plt.title(systemname + " IB ")
                                plt.ylabel("Equity")
                                plt.savefig('./data/results/paper_ib' + systemname + '.png')
                                plt.close(fig)
                                
                                html = html + '<td><img src="paper_ib' + systemname + '.png" width=' + str(width) + ' height=' + str(height) + '><br></td>'
                                
                        counter = counter + 1
                        if counter == 3:
                            html = html + '</tr>'
                            counter=0
    
                        if counter == 0:
                            html = html + '<tr>'
    
    

html = html + '</body></html>'
f = open('./data/results/index.html', 'w')
f.write(html)
f.close()

    #adj_size(model, system['System'],system['Name'],pricefeed,\
    #    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
    #        system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
    #time.sleep(1)



