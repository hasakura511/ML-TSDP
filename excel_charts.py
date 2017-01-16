import time
import math
import numpy as np
import pandas as pd
import sqlite3
from pandas.io import sql
from os import listdir
from os.path import isfile, join
import calendar
import io
import traceback
import json
import re
import datetime
from datetime import datetime as dt
import time
import os
import os.path
import sys
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator,MO, TU, WE, TH, FR, SA, SU,\
                                            MonthLocator, MONDAY, HourLocator, date2num
if len(sys.argv)==1:
    debug=True
else:
    debug=False
    
if debug:
    mode = 'replace'
    #marketList=[sys.argv[1]]
    showPlots=False
    dbPath='./data/futures.sqlite3' 
    dbPath2='D:/ML-TSDP/data/futures.sqlite3' 
    dbPathWeb = 'D:/ML-TSDP/web/tsdp/db.sqlite3'
    dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
    savePath=jsonPath= './data/results/' 
    pngPath = './data/results/' 
    feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
    #test last>old
    #dataPath2=pngPath
    #signalPath = './data/signals/' 
    
    #test last=old
    dataPath2='D:/ML-TSDP/data/'
    
    #signalPath ='D:/ML-TSDP/data/signals2/'
    signalPath ='D:/ML-TSDP/data/signals2/' 
    signalSavePath = './data/signals/' 
    systemPath = './data/systems/' 
    readConn = sqlite3.connect(dbPath2)
    writeConn= sqlite3.connect(dbPath)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='C:/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
else:
    mode= 'replace'
    #marketList=[sys.argv[1]]
    showPlots=False
    feedfile='./data/systems/system_ibfeed.csv'
    dbPath='./data/futures.sqlite3'
    dbPathWeb ='./web/tsdp/db.sqlite3'
    jsonPath ='./web/tsdp/'
    dataPath='./data/csidata/v4futures2/'
    #dataPath='./data/csidata/v4futures2/'
    dataPath2='./data/'
    savePath='./data/results/'
    signalPath = './data/signals2/' 
    signalSavePath = './data/signals2/' 
    pngPath =  './web/tsdp/betting/static/images/'
    systemPath =  './data/systems/'
    readConn = writeConn= sqlite3.connect(dbPath)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
    
readWebConn = sqlite3.connect(dbPathWeb)

active_symbols={
                        'v4futures':['AD', 'BO', 'BP', 'C', 'CD', 'CL', 'CU', 'EMD', 'ES', 'FC',
                                           'FV', 'GC', 'HG', 'HO', 'JY', 'LC', 'LH', 'MP', 'NE', 'NG',
                                           'NIY', 'NQ', 'PA', 'PL', 'RB', 'S', 'SF', 'SI', 'SM', 'TU',
                                           'TY', 'US', 'W', 'YM'],
                        'v4mini':['C', 'CL', 'CU', 'EMD', 'ES', 'HG', 'JY', 'NG', 'SM', 'TU', 'TY', 'W'],
                        'v4micro':['BO', 'ES', 'HG', 'NG', 'TY'],
                        }
                        
def saveCharts(df, **kwargs):
    ylabel=kwargs.get('ylabel','')
    xlabel=kwargs.get('xlabel','')
    title=kwargs.get('title','')
    filename=kwargs.get('filename','chart.png')
    width=kwargs.get('width',0.8)
    legend_outside=kwargs.get('legend_outside',False)
    figsize=kwargs.get('figsize',(15,13))
    kind=kwargs.get('kind','bar')
    font = {
            'weight' : 'normal',
            'size'   : 22}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=figsize)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, lookback)])

    df.plot(kind=kind, ax=ax, width=width)
    if legend_outside:
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.15),prop={'size':24},
          fancybox=True, shadow=True, ncol=3)
    else:
        plt.legend(loc='best', prop={'size':24})
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.ylabel(ylabel, size=24)
    #plt.ylabel('Cumulative %change', size=12)
    plt.xlabel(xlabel, size=24)
    #plt.xticks(range(nrows), benchmark_xaxis_label)
    #fig.autofmt_xdate()
    plt.title(title, size=30)
    
    #filename2=date+'_'+account+'_'+line.replace('/','')+'.png'
    plt.savefig(filename, bbox_inches='tight')
    print 'Saved',filename
    if debug:
        plt.show()
    plt.close()

#for account in active_symbols:

account='v4futures'
totalsDF=pd.read_sql('select * from {}'.format('totalsDF_board_'+account), con=readConn,  index_col='Date')
totalsDF.index=totalsDF.index.astype(str).to_datetime().strftime('%A, %b %Y')

#volatility by group
lookback=5
volatility_cols=[x for x in totalsDF.columns if 'Vol' in x and 'Total' not in x]
df=totalsDF[volatility_cols].iloc[-lookback:].transpose()
ylabel='$ Volatiliy'
xlabel='Group'
title=str(lookback)+' Day Volatility by Group'
filename=pngPath+account+'_'+title.replace(' ','')+'.png'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename)

#change by group
lookback=5
change_cols=[x for x in totalsDF.columns if 'Chg' in x and 'Total' not in x]
df=totalsDF[change_cols].iloc[-lookback:].transpose()
ylabel='$ Change'
xlabel='Group'
title=str(lookback)+' Day Change by Group'
filename=pngPath+account+'_'+title.replace(' ','')+'.png'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename)

#long% columns
lookback=5
lper_cols=['L%_currency','L%_energy','L%_grain','L%_index','L%_meat','L%_metal','L%_rates','L%_Total',]
df=totalsDF[lper_cols].iloc[-lookback:].transpose()*100
ylabel='% Long'
xlabel='Group'
title=str(lookback)+' Day Long % by Group'
filename=pngPath+account+'_'+title.replace(' ','')+'.png'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, legend_outside= True)

#accuracy
lookback=3
acc_cols=[x for x in totalsDF.columns if 'ACC' in x and 'benchmark' not in x and 'Off' not in x]
df=totalsDF[acc_cols].iloc[-lookback:].transpose().sort_values(by=df.columns[-1])*100
ylabel='Systems'
xlabel='% Accuracy'
title=str(lookback)+' Day % Accuracy by Group'
filename=pngPath+account+'_'+title.replace(' ','')+'.png'
width=.6
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, figsize=(15,50), kind='barh',width=width)

#accuracy
lookback=3
pnl_cols=[x for x in totalsDF.columns if 'PNL' in x and 'benchmark' not in x and 'Off' not in x]
df=totalsDF[pnl_cols].iloc[-lookback:].transpose().sort_values(by=df.columns[-1])*100
ylabel='Systems'
xlabel='$ PNL'
title=str(lookback)+' Day $ PNL by Group'
filename=pngPath+account+'_'+title.replace(' ','')+'.png'
width=.6
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, figsize=(15,50), kind='barh',width=width)
