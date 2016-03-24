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
        
def generate_paper_ib_plot(systemname, dateCol):
    filename='./data/paper/ib_' + systemname + '_account.csv'
    dataSet=pd.DataFrame([[20000,20000, '2016-01-01']], columns=['equitycurve','PurePLcurve',dateCol])

    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        dataSet=dataSet.sort_values(by=[dateCol])
        dataSet['equitycurve'] = dataSet['balance']
        dataSet['PurePLcurve'] = dataSet['purebalance']
        
    SST=dataSet
    SST[dateCol]=pd.to_datetime(SST[dateCol])
    SST=SST.sort_values(by=[dateCol])
    SST=SST.set_index(dateCol)
    return SST
    
def view_plot(colnames, title, ylabel, SST):
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
    plt.show()
    #plt.savefig(filename)
    plt.close(fig)
    plt.close()
    
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
    