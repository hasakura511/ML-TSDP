import numpy as np
import pandas as pd
import time
import ibapi.get_feed as feed
import time
import os
import logging
import threading
import sys

"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
logging.basicConfig(filename='/logs/get_hist.log',level=logging.DEBUG)

dataPath = './data/from_IB/'
minDataPoints = 10000
durationStr='2 W'
barSizeSetting='30 min'
whatToShow='MIDPOINT'

def start_feed(symFilter):
    feed.cache_bar_csv(dataPath, barSizeSetting, symFilter)
    
    if durationStr == '1 min':
        threads = []
        feed_thread = threading.Thread(target=feed.get_bar_feed, args=[dataPath, whatToShow, barSizeSetting, symFilter])
        feed_thread.daemon=True
        threads.append(feed_thread)
        [t.start() for t in threads]
    data=feed.get_bar_hist(dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symFilter)
    interval=feed.duration_to_interval(barSizeSetting)
    filename=dataPath+interval+'_'+symFilter+'.csv'
    data.to_csv(filename)
    #[t.join() for t in threads]

if len(sys.argv) > 1:
    symFilter=sys.argv[1]
    start_feed(symFilter)


