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
import seitoolz.bars as bars
from multiprocessing import Process, Queue
import os
from pytz import timezone
from dateutil.parser import parse


    


def get_last_bars_debug(currencyPairs, ylabel, callback, **kwargs):
    global tickerId
    global lastDate
    minPath = kwargs.get('minPath','./data/bars/')
    #while 1:
    try:
        SST=pd.DataFrame()
        symbols=list()
        returnData=False
        for i,ticker in enumerate(currencyPairs):
            pair=ticker
            minFile=minPath+pair+'.csv'
            symbol = pair
            #print minFile
            if os.path.isfile(minFile):
                logging.info(str(i)+' loading '+minFile)
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
                
                #if len(SST.index.values) < 1:
                #    SST=dta
                #else:
                 #   SST=SST.join(dta)
                    
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
        
def get_bars(pairs, interval):
    #global SST
    #global start_time
    mypairs=list()
    for pair in pairs:
        mypairs.append(interval + pair)
        
    if debug:
        get_last_bars_debug(mypairs, 'Close', onBar,minPath=minPath)
    else:
        bars.get_last_bars(mypairs, 'Close', onBar)
        #get_last_bars_debug(mypairs, 'Close', onBar)

def onBar(bar, symbols):
    global start_time
    global gotbar
    global pairs
    bar = bar.iloc[-1]
    logging.info('received '+str(symbols)+str(bar))
    if not gotbar.has_key(bar['Date']):
        gotbar[bar['Date']]=list()
    #print bar['Date'], gotbar[bar['Date']]
    for symbol in symbols:
        if symbol not in gotbar[bar['Date']]:
            gotbar[bar['Date']].append(symbol)
    #gotbar[bar['Date']]=[i for sublist in gotbar[bar['Date']] for i in sublist]
    logging.info(str(bar['Date'])+ str(gotbar[bar['Date']])+ str(len(gotbar[bar['Date']]))+'bars '+ str(len(pairs))+'pairs')
    #global SST
    #SST = SST.combine_first(bar).sort_index()
    #if debug:
    
    if len(gotbar[bar['Date']])==len(livePairs):
    #if len([p for p in gotbar[bar['Date']] if p in livePairs]) == len(livePairs):
        #print gotbar[bar['Date']]
        start_time2 = time.time()
        for sym in gotbar[bar['Date']]:
            logging.info('')
            logging.info(sym+' timenow: '+dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z"))
            runPair_v1([sym])
        logging.info( 'All signals created for bar '+str(bar['Date']))
        logging.info('Runtime: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes' ) 
        logging.info('Last bar time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes\n' ) 
        start_time = time.time()
    #else:   
    #    if len(gotbar[bar['Date']])==len(pairs):
    #        print  len(gotbar[bar['Date']]), 'bars collected for', bar['Date'],'running systems..'
     #       for sym in gotbar[bar['Date']]:
                #print sym
     #           runPair_v1(sym)
                
    
        
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
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData,\
                'modelPath':modelPath,'loadModel':loadModel}
                
        
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
            logging.info('v1 '+' signal '+str(signal.signals)+ ' safef '+str(signal.safef)+' CAR25 '+str(signal.CAR25))
            logging.info(signal.system)
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
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData,\
                'modelPath':modelPath,'loadModel':loadModel}
                

        
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
            if dataSet is not None:
                signal, dataSet=runv2(runData, dataSet)
            else:
                signal, dataSet=runv2(runData)
            print signal
            logging.info('v2 '+' signal '+str(signal.signals)+ ' safef '+str(signal.safef)+' CAR25 '+str(signal.CAR25))
            logging.info(signal.system)
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
start_time = time.time()
lastDate={}
tickerId=1
gotbar=dict()

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='30m'

pairs =  [
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
                 
if len(sys.argv)==1:          
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
    debug=True
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    #runDPS = True
    saveParams = False
    saveDataSet=True
    verbose= False
    loadModel=True
    #paths
    scorePath = None
    equityStatsSavePath = None
    #scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    #equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    modelPath = 'D:/ML-TSDP/data/models/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/' 
    minPath= 'D:/ML-TSDP/data/bars/'
    #while 1:          
    get_bars(livePairs,barSizeSetting+'_')
    #time.sleep(100)
    
    
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
    loadModel=False
    #paths
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    modelPath = './data/models/'
    bestParamsPath =  './data/params/'
    chartSavePath = './data/results/' 
    minPath= './data/bars/'
    
    if len(sys.argv) >2:
        if sys.argv[2] == 'debug':  
            debug = True
            logging.info( 'running debug mode...' )
        
    if sys.argv[1] == 'single':  
        while 1:
            start_time = time.time()
            logging.info( 'starting single thread mode for '+str(barSizeSetting)+' bars '+str(len(livePairs))+' pairs.')
            logging.info(str(livePairs) )
            get_bars(livePairs,barSizeSetting+'_')
            #time.sleep(100)
    elif sys.argv[1] == 'multi':
        runThreads()
    else:
        #print 'please specify single or multi, thanks.'
        sys.exit('please specify single or multi.')
    

