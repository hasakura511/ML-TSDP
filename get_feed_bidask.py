import numpy as np
import pandas as pd
import ibapi.get_feed as feed
import time
import logging

logging.basicConfig(filename='/logs/get_feed_bidask.log',level=logging.DEBUG)
  
def start_bidask_feed():
      feed.get_bar_bidask()
      
      while 1:
          time.sleep(100)
         
start_bidask_feed()
    

