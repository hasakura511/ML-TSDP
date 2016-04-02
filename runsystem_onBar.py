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
import sys
import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
import websocket

logging.basicConfig(filename='/logs/runsystem_v1.log',level=logging.DEBUG)

debug=False

pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                
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
        runPair(sym)
        
def runPair(pair):
    try:
            f=open ('/logs/' + pair + 'onBar.log','a')
            print 'Starting V1: ' + pair
            f.write('Starting V1: ' + pair)
            
            ferr=open ('/logs/' + pair + 'onBar_err.log','a')
            ferr.write('Starting V1: ' + pair)
            
            subprocess.call(['python','debug_system_v1.3C_30min.py',pair,'1'], stdout=f, stderr=ferr)
            f.close()
            ferr.close()
     except Exception as e:
        	 #f=open ('./debug/v1run' + pair + '.log','a')
        	 #f.write(e)
        	 #f.close()
        	 logging.error("something bad happened", exc_info=True)
    
threads = []
for pair in pairs:
	sig_thread = threading.Thread(target=get_bars, args=[[pair], '30m_'])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
