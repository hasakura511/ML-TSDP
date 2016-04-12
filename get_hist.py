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
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'

def start_feed(symFilter):
    feed.cache_bar_csv(dataPath, barSizeSetting, symFilter)
    
    threads = []
    feed_thread = threading.Thread(target=feed.get_bar_feed, args=[dataPath, whatToShow, barSizeSetting, symFilter])
    feed_thread.daemon=True
    threads.append(feed_thread)
    
    hist_thread = threading.Thread(target=feed.get_bar_hist, args=[dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symFilter])
    hist_thread.daemon=True
    threads.append(hist_thread)
    
    [t.start() for t in threads]
    [t.join() for t in threads]

if len(sys.argv) > 1:
    symFilter=sys.argv[1]
    start_feed(symFilter)


