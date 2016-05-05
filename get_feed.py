import numpy as np
import pandas as pd
import time
import ibapi.get_feed as feed
import time
import os
import logging
import threading

"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
logging.basicConfig(filename='/logs/get_feed_1m.log',level=logging.DEBUG)

dataPath = './data/from_IB/'
minDataPoints = 10000
durationStr='1 D'
barSizeSetting='1 min'
whatToShow='MIDPOINT'

def get_ibfeed(contract, tickerId):
	get_feed(contract, tickerId)

        
def check_bar():
    finished=False
    time.sleep(120)
    while not finished:
        try:
            has_feed=feed.check_bar(barSizeSetting)
            if not has_feed:
                logging.error('Feed not being received - restarting')
                feed.reconnect_ib()
                start_feed()
                time.sleep(120)
            time.sleep(30)
        except Exception as e:
            logging.error("check_bar", exc_info=True)
            
def start_feed():
    feed.cache_bar_csv(dataPath, barSizeSetting)
    
    threads = []
    feed_thread = threading.Thread(target=feed.get_bar_feed, args=[dataPath, whatToShow, barSizeSetting])
    feed_thread.daemon=True
    threads.append(feed_thread)
    
    hist_thread = threading.Thread(target=feed.get_bar_hist, args=[dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting])
    hist_thread.daemon=True
    threads.append(hist_thread)
    
    [t.start() for t in threads]
    #[t.join() for t in threads]


start_feed()
check_bar()

