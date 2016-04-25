import seitoolz.bars as bars
import pandas as pd
import threading
import time
import logging
import ibapi.get_feed as feed
from pytz import timezone
from dateutil.parser import parse
import datetime

logging.basicConfig(filename='/logs/get_feed_30m.log',level=logging.DEBUG)

interval='30m'
minDataPoints = 10000

def get_history(contracts):
    global interval
    global minDataPoints
    dataPath = './data/from_IB/'
    
    (durationStr, barSizeSetting, whatToShow)=feed.interval_to_ibhist_duration(interval)
    feed.cache_bar_csv(dataPath, barSizeSetting)
    for contract in contracts:
        try:
            histdata = feed.get_bar_hist(dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter='')
            bars.proc_history(contract, histdata, interval)
        except Exception as e:
            logging.error("something bad happened", exc_info=True)
    
    while 1:
        dataSet=pd.read_csv('./data/systems/restore_hist.csv', index_col=0)
        for date in dataSet.index:
            try:
                eastern=timezone('US/Eastern')
                #timestamp
                date=parse(str(date)).replace(tzinfo=eastern)
                for contract in contracts:
                  
                        histdata = feed.get_bar_hist_date(date, dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter='')
                        bars.proc_history(contract, histdata, interval)
            except Exception as e:
<<<<<<< HEAD
                logging.error("something bad happened", exc_info=True)  
=======
                logging.error("something bad happened", exc_info=True)
        dataSet=pd.DataFrame({}, columns=['Date'])
        dataSet.to_csv('./data/systems/restore_hist.csv')
>>>>>>> 5eb591542d5e8ba98b189224ac54c6669e977ecb
        time.sleep(600)    
         
def start_proc():
    global interval
    
    contracts=bars.get_contracts()
    pairs=bars.get_symbols()
    threads = []
    
    t1 = threading.Thread(target=get_history, args=[contracts])
    t1.daemon=True
    threads.append(t1)
    
    [t.start() for t in threads]
    #[t.join() for t in threads]
    bars.update_bars(pairs, interval)
    

start_proc()


