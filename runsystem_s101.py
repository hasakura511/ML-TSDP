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

logging.basicConfig(filename='/logs/runsystem_v2.log',level=logging.DEBUG)

debug=False

pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                
def runv2(pair):
      #while 1:
      try:
            f=open ('/logs/s101v2' + pair + 'v2.log','a')
            print 'Starting s101v2: ' + pair
            f.write('Starting s101v2: ' + pair)
            
            ferr=open ('/logs/s101' + pair + 'v2_err.log','a')
            ferr.write('Starting s101: ' + pair)
            subprocess.call(['python','system_s101.py','2','2','0',pair,'1'], stdout=f, stderr=ferr)
            
            subprocess.call(['python','system_s101.py','9','2','0',pair], stdout=f, stderr=ferr)
            f.close()
            ferr.close()
      except Exception as e:
        	 #f=open ('./debug/v2run' + pair + '.log','a')
        	 #f.write(e)
        	 #f.close()
        	 logging.error("something bad happened", exc_info=True)
      return

threads = []
for pair in pairs:
	sig_thread = threading.Thread(target=runv2, args=[pair])
	sig_thread.daemon=True
	threads.append(sig_thread)
[t.start() for t in threads]
[t.join() for t in threads]

	
