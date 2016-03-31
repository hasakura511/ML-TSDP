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

logging.basicConfig(filename='/logs/runproc_v3.log',level=logging.DEBUG)

debug=False

files=[['get_results.py',['sig'],1200],
       ['get_results.py',['c2'],40000],
       ['get_results.py',['c2_2'],25000],
       ['get_results.py',['ib'],43200],
       ['get_results.py',['paper'],12000],
       ['get_results.py',['paper2'],1200],
       ['get_results.py',['btc'], 20500]]

def runfile(file,runargs,sleeptime):
    while 1:
     try:
        time.sleep(sleeptime)
        f=open ('/logs/' + file + '.log','a')
        print 'Starting ' + file
        f.write('Starting : ' + file)
         
        ferr=open ('/logs/' + file + '_err.log','a')
        ferr.write('Starting : ' + file)
        
        runproc=['python',file]
        runproc=runproc + runargs
        for proc in runproc:
          print ' Arg: ' + proc
        
        subprocess.call(runproc, stdout=f, stderr=ferr)

        f.close()
        ferr.close()
     except Exception as e:
	 #f=open ('./debug/v3run' + file + '.log','a')
	 #f.write(e)
	 #f.close()
	 logging.error("Exception: " + str(file), exc_info=True)
    return
    
threads = []
for (file,runargs,sleeptime) in files:
	sig_thread = threading.Thread(target=runfile, args=[file,runargs,sleeptime])
	sig_thread.daemon=True
	threads.append(sig_thread)
	sig_thread.start()
while 1:
	time.sleep(100)
	
