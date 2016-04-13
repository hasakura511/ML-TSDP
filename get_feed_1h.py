import seitoolz.bars as bars
import threading
import time
import logging
import ibapi.get_feed as feed

logging.basicConfig(filename='/logs/get_feed_1h.log',level=logging.DEBUG)

interval='1h'
minDataPoints = 10000

def get_history(contracts):
    global interval
    global minDataPoints
    dataPath = './data/from_IB/'
    
    (durationStr, barSizeSetting, whatToShow)=feed.interval_to_ibhist_duration(interval)
    feed.cache_bar_csv(dataPath, barSizeSetting)
    for contract in contracts:
        histdata = feed.get_bar_hist(dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter='')
        bars.proc_history(contract, histdata, interval)
        
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


