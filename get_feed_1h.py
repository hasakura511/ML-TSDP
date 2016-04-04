import seitoolz.bars as bars
import threading
import time
import logging

logging.basicConfig(filename='/logs/get_feed_1h.log',level=logging.DEBUG)

def start_proc():
    interval='1h'
    minDataPoints = 5000
    exchange='IDEALPRO'
    secType='CASH'
    
    pairs=bars.get_currencies()
    threads = []
    #bars.get_hist_bars(pairs, interval, minDataPoints, exchange, secType)
    #bars.create_bars(pairs, interval)
      
    t1 = threading.Thread(target=bars.get_hist_bars, args=[pairs, interval, minDataPoints, exchange, secType])
    t1.daemon=True
    threads.append(t1)
    
    #t2 = threading.Thread(target=bars.create_bars, args=[pairs, interval])
    #t2.daemon=True
    #threads.append(t2)
    
    [t.start() for t in threads]
    #[t.join() for t in threads]
    bars.update_bars(pairs, interval)
    

start_proc()


