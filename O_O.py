import socket   
import select
import sys
import pytz
from os import listdir
from os.path import isfile, join
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
import sys
#import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
#import websocket
'''
import copy
currencyPairs =   [
                'NZDJPY',\
                'CADJPY',\
                'CHFJPY',\
                'EURJPY',\
                'GBPJPY',\
                'AUDJPY',\
                'USDJPY',\
                'AUDUSD',\
                'EURUSD',\
                'EURAUD',\
                'EURCAD',\
                'EURNZD',\
                'GBPUSD',\
                'USDCAD',\
                'USDCHF',\
                'NZDUSD',
                'EURCHF',\
                'EURGBP',\
                'AUDCAD',\
                'AUDCHF',\
                'AUDNZD',\
                'GBPAUD',\
                'GBPCAD',\
                'GBPNZD',\
                'GBPCHF',\
                'CADCHF',\
                'NZDCHF',\
                'NZDCAD'
                ]
ranking = ['AUD','EUR','NZD','CHF','GBP','CAD','USD','JPY']
buyHold=[]
sellHold=[]
cplist = copy.deepcopy(currencyPairs)
for currency in ranking:
    for i,pair in enumerate(cplist):
        #print pair
        if pair not in buyHold and pair not in sellHold:
            if currency in pair[0:3]:
                #print i,'bh',pair
                buyHold.append(pair)
                #cplist.remove(pair)
            elif currency in pair[3:6]:
                #print i,'sh',pair
                sellHold.append(pair)
                #cplist.remove(pair)
            #else:
                #print i,currency,pair
print 'buyHold',len(buyHold),buyHold
print 'sellHold',len(sellHold),sellHold
'''

offlinePairs=[]
buyHold=['GBPJPY', 'GBPUSD', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'GBPCHF', 'CADCHF', 'EURUSD', 'EURAUD', 'EURNZD', 'EURCHF']
sellHold=['EURGBP', 'NZDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'AUDJPY', 'USDJPY', 'EURCAD', 'USDCAD', 'AUDCAD', 'NZDCAD', 'USDCHF', 'AUDCHF', 'NZDCHF', 'AUDUSD', 'NZDUSD', 'AUDNZD']

 
start_time3 = time.time()
signalPath = './data/signals/'
def offlineMode(ticker, errorText, signalPath, ver1, ver2):
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if 'v'+ver1+'_'+ ticker + '.csv' in files:
            signalFile=pd.read_csv(signalPath+ 'v'+ver1+'_'+ ticker + '.csv', parse_dates=['dates'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.signals = 0
            offline.gainAhead =0
            offline.prior_index=0
            offline.safef=0
            offline.CAR25=0
            offline.dd95 = 0
            offline.ddTol=0
            offline.system = errorText
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + 'v'+ver1+'_'+ ticker + '.csv', index=False)
            
        if 'v'+ver2+'_'+ ticker + '.csv' in files:
            signalFile=pd.read_csv(signalPath+ 'v'+ver2+'_'+ ticker + '.csv', parse_dates=['dates'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.signals = 0
            offline.gainAhead =0
            offline.prior_index=0
            offline.safef=0
            offline.CAR25=0
            offline.dd95 = 0
            offline.ddTol=0
            offline.system = errorText
            offline.timestamp = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.cycleTime = 0
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + 'v'+ver2+'_'+ ticker + '.csv', index=False)
        print errorText    
        #sys.exit(errorText)
        
logging.basicConfig(filename='/logs/runsystem_v3v4.log',level=logging.DEBUG)
############v4 OFFLINE##########
version = '4'
version_ = '43'
barSize='30m'
bias='sellHold'
adfPvalue='-3'
validationSetLength ='1'
useSignalsFrom='tripleFiltered'
offline=True
scriptName= 'debug_system_v'+version_+'C_30min.py'
pairs=offlinePairs
#pairs=['EURCHF','EURUSD','USDCHF','EURCAD','GBPCAD']
#pairsList=[pairs,pairs2,pairs3]
#logging.basicConfig(filename='/logs/runsystem_v'+version+'_'+bias+'.log',level=logging.DEBUG)

        
#def runv4(pairs):
#while 1:
start_time = time.time()
for pair in pairs:
    try:
        start_time2 = time.time()
        logging.info(str(dt.now())+' running v'+version_+' '+pair)
        f=open ('/logs/' + pair + 'v'+version+ '.log','a')
        print str(dt.now()), ' Starting v'+version_+': ' + pair
        f.write(str(dt.now())+' Starting v'+version_+': ' + pair)

        ferr=open ('/logs/' + pair + 'v'+version+'_err.log','a')
        ferr.write( str(dt.now())+' Starting v'+version_+': ' + pair)


        if offline:
            message = "Offline Mode: turned off in runsystem"
            offlineMode(pair, message, signalPath, version, version_)
            logging.info('v'+version+' '+pair+' '+message)
        else:
            subprocess.call(['python',scriptName,pair,bias,adfPvalue,validationSetLength,useSignalsFrom],\
                                stdout=f, stderr=ferr)
        
        f.close()
        ferr.close()
        signal=pd.read_csv('./data/signals/v'+version+'_'+pair+'.csv').iloc[-1]
        logging.info('Check signal:' +str(signal.signals))
        logging.info('Elapsed time: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes. Time now '+\
                            dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z")) 
    except Exception as e:
        #f=open ('./debug/v4run' + pair + '.log','a')
        #f.write(e)
        #f.close()
        logging.error("something bad happened", exc_info=True)
        #return
logging.info(str(len(pairs))+' pairs completed v'+version_+' '+barSize+' '+'OFFLINE')
logging.info('Offline Cycle time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes' ) 
print len(pairs), 'pairs completed', barSize, 'OFFLINE'
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'


#################################################
version = '4'
version_ = '43'
barSize='30m'
bias='buyHold'
adfPvalue='-3'
validationSetLength ='1'
useSignalsFrom='tripleFiltered'
offline=False
scriptName= 'debug_system_v'+version_+'C_30min.py'
pairs=buyHold

#pairsList=[pairs,pairs2,pairs3]
#logging.basicConfig(filename='/logs/runsystem_v'+version+'_'+bias+'.log',level=logging.DEBUG)

        
#def runv4(pairs):
#while 1:
start_time = time.time()
for pair in pairs:
    try:
        start_time2 = time.time()
        logging.info(str(dt.now())+' running v'+version_+' '+pair)
        f=open ('/logs/' + pair + 'v'+version+ '.log','a')
        print str(dt.now()), ' Starting v'+version_+': ' + pair
        f.write(str(dt.now())+' Starting v'+version_+': ' + pair)

        ferr=open ('/logs/' + pair + 'v'+version+'_err.log','a')
        ferr.write( str(dt.now())+' Starting v'+version_+': ' + pair)

        if offline:
            message = "Offline Mode: turned off in runsystem"
            offlineMode(pair, message, signalPath, version, version_)
            logging.info('v'+version+' '+pair+' '+message)
        else:
            subprocess.call(['python',scriptName,pair,bias,adfPvalue,validationSetLength,useSignalsFrom],\
                                stdout=f, stderr=ferr)        
        f.close()
        ferr.close()
        signal=pd.read_csv('./data/signals/v'+version+'_'+pair+'.csv').iloc[-1]
        logging.info(str(signal))
        logging.info('Elapsed time: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes. Time now '+\
                            dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z")) 
    except Exception as e:
        #f=open ('./debug/v4run' + pair + '.log','a')
        #f.write(e)
        #f.close()
        logging.error("something bad happened", exc_info=True)
        #return
logging.info(str(len(pairs))+' pairs completed v'+version_+' '+barSize+' '+bias)
logging.info('Cycle time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes' ) 
print len(pairs), 'pairs completed', barSize, bias, adfPvalue, scriptName,useSignalsFrom
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'

#################################################
version = '4'
version_ = '43'
barSize='30m'
bias='sellHold'
adfPvalue='-3'
validationSetLength ='1'
useSignalsFrom='tripleFiltered'
offline=False
scriptName= 'debug_system_v'+version_+'C_30min.py'
pairs=sellHold
            
#logging.basicConfig(filename='/logs/runsystem_v'+version+'_'+bias+'.log',level=logging.DEBUG)

        
#def runv4(pairs):
#while 1:
start_time = time.time()
for pair in pairs:
    try:
        start_time2 = time.time()
        logging.info(str(dt.now())+' running v'+version_+' '+pair)
        f=open ('/logs/' + pair + 'v'+version+ '.log','a')
        print str(dt.now()), ' Starting v'+version_+': ' + pair
        f.write(str(dt.now())+' Starting v'+version_+': ' + pair)

        ferr=open ('/logs/' + pair + 'v'+version+'_err.log','a')
        ferr.write( str(dt.now())+' Starting v'+version_+': ' + pair)


        if offline:
            message = "Offline Mode: turned off in runsystem"
            offlineMode(pair, message, signalPath, version, version_)
            logging.info('v'+version+' '+pair+' '+message)
        else:
            subprocess.call(['python',scriptName,pair,bias,adfPvalue,validationSetLength,useSignalsFrom],\
                                stdout=f, stderr=ferr)
        
        f.close()
        ferr.close()
        signal=pd.read_csv('./data/signals/v'+version+'_'+pair+'.csv').iloc[-1]
        logging.info(str(signal))
        logging.info('Elapsed time: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes. Time now '+\
                            dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z")) 
    except Exception as e:
        #f=open ('./debug/v4run' + pair + '.log','a')
        #f.write(e)
        #f.close()
        logging.error("something bad happened", exc_info=True)
        #return
logging.info(str(len(pairs))+' pairs completed v'+version_+' '+barSize+' '+bias)
logging.info('Cycle time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes' ) 
print len(pairs), 'pairs completed', barSize, bias, adfPvalue, scriptName, useSignalsFrom
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
print 'Total Elapsed time: ', round(((time.time() - start_time3)/60),2), ' minutes'


'''
threads = []
for list in pairsList:
	logging.info(str(list))
	sig_thread = threading.Thread(target=runv4, args=list)
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
'''
	
