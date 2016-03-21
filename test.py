import socket   
import select
import sys
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone

import json
import time
from pandas.io.json import json_normalize
import pandas as pd
import threading
from btapi.get_signal import get_v1signal
from btapi.get_hist_btcharts import get_bthist
from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
from seitoolz.paper import adj_size
import re
from os import listdir
from os.path import isfile, join

dataPath='./data/btapi/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('BTCUSD')
debug=False

if len(sys.argv)==1:
    debug=True
for file in files:
        if re.search(btcsearch, file):
                systemname=file
                systemname = re.sub(dataPath + 'BTCUSD_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
		print systemname
		if systemname == 'BTCUSD_rockEUR':
                 data = pd.read_csv(dataPath + file, index_col='Date')
                 if data.shape[0] > 2000:
                    #get_v1signal(data, 'BTCUSD_' + systemname, systemname, debug)
                    get_v1signal(data.tail(2000), 'BTCUSD_' + systemname, systemname, debug)
