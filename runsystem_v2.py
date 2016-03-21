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

logging.basicConfig()

debug=False

pairs=['NZDJPY','CADJPY','CHFJPY','EURGBP',\
                 'GBPJPY','EURCHF','AUDJPY',\
                 'AUDUSD','EURUSD','GBPUSD','USDCAD',\
                 'USDCHF','USDJPY','EURJPY','NZDUSD']
                

def runv2(pair):
    while 1:
        print 'Starting V2: ' + pair
        f=open ('/logs/' + pair + 'v2.log','a')
        subprocess.call(['python','system_v2.31C.py',pair,'1'], stdout=f)
        f.close()
    return

threads = []
for pair in pairs:
	sig_thread = threading.Thread(target=runv2, args=[pair])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
