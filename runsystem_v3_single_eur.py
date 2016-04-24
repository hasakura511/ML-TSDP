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
import sys
#import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
#import websocket

logging.basicConfig(filename='/logs/runsystem_v3_eur.log',level=logging.DEBUG)
version_ = '3.1'
barSize='30m'
debug=False
pairs=['EURAUD','EURNZD','EURCHF','GBPUSD','USDCAD','USDCHF','GBPNZD','GBPCAD','GBPAUD','AUDCAD','CADCHF','AUDNZD']
#pairs=['EURCAD','EURAUD','EURNZD','EURCHF','GBPUSD','USDCAD','USDCHF']
#pairs=['EURCAD','EURAUD','EURNZD','EURGBP','EURCHF','EURUSD','GBPUSD','USDCAD']
#pairs=['EURCAD','EURAUD','USDCAD','EURNZD','EURGBP','USDCHF','EURCHF']
#pairs=['EURGBP','EURCAD','EURAUD','EURNZD','USDCAD']
#pairs=['AUDUSD','EURUSD','GBPUSD','USDCAD','NZDUSD','USDCHF']
#pairs=['NZDJPY','CADJPY','CHFJPY','USDJPY','GBPJPY','EURJPY','AUDJPY']
#pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP','GBPJPY']
#pairs=['EURCHF','AUDJPY','AUDUSD','EURUSD','GBPUSD']
#pairs=['USDCAD','USDCHF','USDJPY','EURJPY','NZDUSD']
                


#def runv3(pair):
#while 1:
start_time = time.time()
for pair in pairs:
    try:
        start_time2 = time.time()
        logging.info('running v3'+pair)
        f=open ('/logs/' + pair + 'v3.log','a')
        print 'Starting v3: ' + pair
        f.write('Starting v3: ' + pair)

        ferr=open ('/logs/' + pair + 'v3_err.log','a')
        ferr.write('Starting V3: ' + pair)

        subprocess.call(['python','debug_system_v3.1C_30min_eur.py',pair,'1'], stdout=f, stderr=ferr)
        f.close()
        ferr.close()
        signal=pd.read_csv('./data/signals/v'+version_+'_'+pair+'_'+barSize+'.csv').iloc[-1]
        logging.info(str(signal))
        logging.info('Elapsed time: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes. Time now '+\
                            dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z")) 
    except Exception as e:
        #f=open ('./debug/v3run' + pair + '.log','a')
        #f.write(e)
        #f.close()
        logging.error("something bad happened", exc_info=True)
        #return
logging.info('Cycle time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes' ) 
print len(pairs), 'pairs completed'
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
'''
threads = []
for pair in pairs:
	sig_thread = threading.Thread(target=runv3, args=[pair])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
'''