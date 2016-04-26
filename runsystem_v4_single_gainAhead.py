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


version = '4'
version_ = '43'
barSize='30m'
bias='gainAhead'
volatility='0.1'
scriptName= 'debug_system_v'+version_+'C_30min.py'
pairs=['EURAUD','EURNZD','EURCAD','EURCHF','EURUSD']
logging.basicConfig(filename='/logs/runsystem_v'+version+'_'+bias+'.log',level=logging.DEBUG)


#def runv4(pair):
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

        subprocess.call(['python',scriptName,pair,bias,volatility],\
                                stdout=f, stderr=ferr)
        f.close()
        ferr.close()
        signal=pd.read_csv('./data/signals/v'+version+'_'+pair+'_'+barSize+'.csv').iloc[-1]
        logging.info(str(signal))
        logging.info('Elapsed time: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes. Time now '+\
                            dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z")) 
    except Exception as e:
        #f=open ('./debug/v4run' + pair + '.log','a')
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
	sig_thread = threading.Thread(target=runv4, args=[pair])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
'''