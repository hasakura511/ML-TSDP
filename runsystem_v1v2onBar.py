import socket   
import select
import sys
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import subprocess
import json
import time
from pandas.io.json import json_normalize
import pandas as pd
import threading
#from btapi.get_signal import get_v1signal
#from btapi.get_hist_btcharts import get_bthist
#from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
#from seitoolz.paper import adj_size
from suztoolz.debug_system_v2_30min_func import runv2
import sys
#import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
#import websocket
from suztoolz.display import offlineMode
#import seitoolz.bars as bars
from multiprocessing import Process, Queue
import os
from pytz import timezone
from dateutil.parser import parse


    


def get_last_bars_debug(currencyPairs, ylabel, callback):
    global tickerId
    global lastDate
    #while 1:
    try:
        SST=pd.DataFrame()
        symbols=list()
        returnData=False
        for ticker in currencyPairs:
            pair=ticker
            minFile='D:/ML-TSDP/data/bars/'+pair+'.csv'
            symbol = pair
            #print minFile
            if os.path.isfile(minFile):
                print 'loading',minFile
                dta=pd.read_csv(minFile).iloc[-1]
                date=dta['Date']
                
                eastern=timezone('US/Eastern')
                date=parse(date).replace(tzinfo=eastern)
                timestamp = time.mktime(date.timetuple())
                dta.Date = date
                #data=pd.DataFrame()
                #data['Date']=date
                #data[symbol]=dta[ylabel]
                #print dta[ylabel]
                #print data, date
                #data=data.set_index('Date') 
                dta.name = dta.Date
                
                if len(SST.index.values) < 1:
                    SST=dta
                else:
                    SST=SST.join(dta)
                    
                if not lastDate.has_key(symbol):
                    lastDate[symbol]=timestamp
                                           
                if lastDate[symbol] < timestamp:
                    returnData=True
                symbols.append(symbol)
                        
            #if returnData:
            #data=SST
            #data=data.set_index('Date')
            #data=data.fillna(method='pad')
                callback(dta, symbols)
            #time.sleep(20)
    except Exception as e:
        logging.error("get_last_bar", exc_info=True)
        
def get_last_bars(currencyPairs, ylabel, callback):
    global tickerId
    global lastDate
    while 1:
        try:
            SST=pd.DataFrame()
            symbols=list()
            returnData=False
            for ticker in currencyPairs:
                pair=ticker
                minFile='./data/bars/'+pair+'.csv'
                symbol = pair
                
                if os.path.isfile(minFile):
                    dta=pd.read_csv(minFile).iloc[-1]
                    date=dta['Date']
                    
                    eastern=timezone('US/Eastern')
                    date=parse(date).replace(tzinfo=eastern)
                    timestamp = time.mktime(date.timetuple())
                    
                    data=pd.DataFrame()
                    data['Date']=date
                    data[symbol]=dta[ylabel]
                    data=data.set_index('Date') 
                    
                    if len(SST.index.values) < 1:
                        SST=data
                    else:
                        SST=SST.join(data)
                        
                    if not lastDate.has_key(symbol):
                        lastDate[symbol]=timestamp
                                               
                    if lastDate[symbol] < timestamp:
                        returnData=True
                        symbols.append(symbol)
                        
            if returnData:
                data=SST
                data=data.set_index('Date')
                data=data.fillna(method='pad')
                callback(data, symbols)
            time.sleep(20)
        except Exception as e:
            logging.error("get_last_bar", exc_info=True)
            
def get_bars(pairs, interval):
    #global SST
    mypairs=list()
    for pair in pairs:
        mypairs.append(interval + pair)
        
    if debug:
        get_last_bars_debug(mypairs, 'Close', onBar)
    else:
        get_last_bars(mypairs, 'Close', onBar)

def onBar(bar, symbols):
    global gotbar
    global pairs
    print symbols
    if not gotbar.has_key(bar['Date']):
        gotbar[bar['Date']]=list()
    #print bar['Date'], gotbar[bar['Date']]
    gotbar[bar['Date']].append(symbols)
    #print bar['Date'], gotbar[bar['Date']], len(gotbar[bar['Date']])
    #global SST
    #SST = SST.combine_first(bar).sort_index()
    if debug:
        #if len(gotbar[bar['Date']])==len(pairs):
            print  len(gotbar[bar['Date']]), 'bars collected for', bar['Date'],'running systems..'
            for sym in gotbar[bar['Date']]:
                print sym,
                runPair_v1(sym)
            print 'All signals created for bar',bar['Date'],'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'    
    else:   
        if len(gotbar[bar['Date']])==len(pairs):
            print  len(gotbar[bar['Date']]), 'bars collected for', bar['Date'],'running systems..'
            for sym in gotbar[bar['Date']]:
                #print sym
                runPair_v1(sym)
                
    
        
def runPair_v1(pair):
    ticker = pair[0].split('_')[1]
    version = 'v1'
    version_ = 'v1.3C'
    runDPS = False
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData}
                
        
    try:
        with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            print 'Starting '+version+': ' + ticker
            if ticker not in livePairs:
                offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
            #f.write('Starting '+version+': ' + ticker)         
            #ferr.write('Starting '+version+': ' + ticker)
            signal, dataSet=runv2(runData)
            print signal
            #subprocess.call(['python','debug_system_v1.3C_30min.py',ticker,'1'], stdout=f, stderr=ferr)
            #f.close()
            #ferr.close()

            sys.stdout = orig_stdout
        runPair_v2(pair, dataSet)
    except Exception as e:
        	 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
        	 #ferr.write(e)
        	 #ferr.close()
        	 logging.error("something bad happened", exc_info=True)
 
def runPair_v2(pair, dataSet):
    ticker = pair[0].split('_')[1]
    version = 'v2'
    version_ = 'v2.4C'
    runDPS = True
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData}
                

        
    try:
        with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            print 'Starting '+version+': ' + ticker
            if ticker not in livePairs:
                offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
            #f.write('Starting '+version+': ' + ticker)
            
            #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
            #ferr.write('Starting '+version+': ' + ticker)
            
            signal, dataSet=runv2(runData, dataSet)
            print signal
            #subprocess.call(['python','debug_system_v1.3C_30min.py',ticker,'1'], stdout=f, stderr=ferr)
            #f.close()
            #ferr.close()
            sys.stdout = orig_stdout
    except Exception as e:
        	 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
        	 #ferr.write(e)
        	 #ferr.close()
        	 logging.error("something bad happened", exc_info=True)
             
def runThreads():
    global start_time
    start_time = time.time()
    threads = []
    for pair in pairs:
        sig_thread = threading.Thread(target=get_bars, args=[[pair], barSizeSetting+'_'])
        sig_thread.daemon=True
        threads.append(sig_thread)
        sig_thread.start()
        
    if debug==False:
        while 1:
            time.sleep(100)



logging.basicConfig(filename='/logs/runsystem_v1v2.log',level=logging.DEBUG)

lastDate={}
tickerId=1
gotbar=dict()

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='30m'

pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                 
if len(sys.argv)==1:          
    livePairs =  [
                    #'NZDJPY',\
                    #'CADJPY',\
                    #'CHFJPY',\
                    #'EURJPY',\
                    #'GBPJPY',\
                    #'AUDJPY',\
                    #'USDJPY',\
                    'AUDUSD',\
                    #'EURUSD',\
                    #'GBPUSD',\
                    #'USDCAD',\
                    #'USDCHF',\
                    #'NZDUSD',
                    #'EURCHF',\
                    #'EURGBP'\
                    ]
    #settings
    debug=True
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    #runDPS = True
    saveParams = False
    saveDataSet=True
    verbose= False
    #paths
    scorePath = None
    equityStatsSavePath = None
    #scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    #equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/' 
    
    #get_bars(['AUDUSD'],'30m_')
    
else:
    livePairs =  [
                    'NZDJPY',\
                    'CADJPY',\
                    'CHFJPY',\
                    'EURJPY',\
                    'GBPJPY',\
                    'AUDJPY',\
                    'USDJPY',\
                    'AUDUSD',\
                    'EURUSD',\
                    'GBPUSD',\
                    'USDCAD',\
                    'USDCHF',\
                    'NZDUSD',
                    'EURCHF',\
                    'EURGBP'\
                    ]
    #settings
    debug=False
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    #runDPS = True
    saveParams = False
    saveDataSet=True
    verbose= False
    #paths
    scorePath = None
    equityStatsSavePath = None
    signalPath = '../data/signals/'
    dataPath = '../data/from_IB/'
    bestParamsPath =  '../data/params/'
    chartSavePath = '../data/results/' 
    
    runThreads()
	
