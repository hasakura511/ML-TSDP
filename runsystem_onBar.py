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
from btapi.get_signal import get_v1signal
from btapi.get_hist_btcharts import get_bthist
from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
from seitoolz.paper import adj_size
from forecast import runv2
import sys
import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
import websocket
from suztoolz.display import offlineMode

logging.basicConfig(filename='/logs/runsystem_v1v2.log',level=logging.DEBUG)

debug=False

filterName = 'DF1'
data_type = 'ALL'
barSizeSetting='30m'

pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                 
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
        

                    
def get_bars(pairs, interval):
    #global SST
    mypairs=list()
    for pair in pairs:
        mypairs.append(interval + pair)
    bars.get_last_bars(mypairs, 'Close', onBar)

def onBar(bar, symbols):
    #global SST
    #SST = SST.combine_first(bar).sort_index()
    for sym in symbols:
        runPair_v1(sym)
        runPair_v2(sym)
        
def runPair_v1(pair):
    ticker = pair.split('_')[1]
    version = 'v1'
    version_ = 'v1.3C'
    runDPS = False
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting}
                
    if ticker not in livePairs:
        offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
        
    try:
            f=open ('/logs/' + version+'_'+pair + 'onBar.log','a')
            print 'Starting '+version+': ' + pair
            f.write('Starting '+version+': ' + pair)
            
            ferr=open ('/logs/' + version+'_'+pair + 'onBar_err.log','a')
            ferr.write('Starting '+version+': ' + pair)
            signal=runv2(runData)
            #subprocess.call(['python','debug_system_v1.3C_30min.py',pair,'1'], stdout=f, stderr=ferr)
            f.close()
            ferr.close()
     except Exception as e:
        	 #f=open ('./debug/v1run' + pair + '.log','a')
        	 #f.write(e)
        	 #f.close()
        	 logging.error("something bad happened", exc_info=True)
 
 def runPair_v2(pair):
    ticker = pair.split('_')[1]
    version = 'v2'
    version_ = 'v2.4C'
    runDPS = True
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting}
                
    if ticker not in livePairs:
        offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
        
    try:
            f=open ('/logs/' + version+'_'+pair + 'onBar.log','a')
            print 'Starting '+version+': ' + pair
            f.write('Starting '+version+': ' + pair)
            
            ferr=open ('/logs/' + version+'_'+pair + 'onBar_err.log','a')
            ferr.write('Starting '+version+': ' + pair)
            
            signal=runv2(runData)
            #subprocess.call(['python','debug_system_v1.3C_30min.py',pair,'1'], stdout=f, stderr=ferr)
            f.close()
            ferr.close()
     except Exception as e:
        	 #f=open ('./debug/v1run' + pair + '.log','a')
        	 #f.write(e)
        	 #f.close()
        	 logging.error("something bad happened", exc_info=True)
             
    
threads = []
for pair in pairs:
	sig_thread = threading.Thread(target=get_bars, args=[[pair], barSizeSetting+'_'])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
