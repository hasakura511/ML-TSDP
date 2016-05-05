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
    #for contract in contracts:
    #    try:
            #histdata = feed.get_bar_hist(dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter='')
            #bars.proc_history(contract, histdata, interval)
    #    except Exception as e:
    #        logging.error("something bad happened", exc_info=True)
    
    while 1:
        try:
            dataSet=pd.read_csv('./data/systems/restore_hist.csv', index_col=0)
            for date in dataSet.index:
            
                eastern=timezone('US/Eastern')
                #timestamp
                mydate=parse(str(date)).replace(tzinfo=eastern)
                for contract in contracts:
                        print mydate, contract.symbol, contract.currency
                        date=feed.get_bar_date(barSizeSetting, mydate) + ' EST'
                        print date
                        histdata = feed.get_bar_hist_date(mydate, dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter='')
                        print histdata                        
                        #bars.proc_history(contract, histdata, interval)
            #dataSet=pd.DataFrame({}, columns=['Date']).set_index('Date')
            #dataSet.to_csv('./data/systems/restore_hist.csv')
            
        except Exception as e:
                logging.error("something bad happened", exc_info=True)
        
        time.sleep(10)    
         
def start_proc():
    global interval
    
    contracts=bars.get_cash_contracts()
    pairs=bars.get_symbols()
    threads = []
    get_history(contracts)
    #t1 = threading.Thread(target=get_history, args=[contracts])
    #t1.daemon=True
    #threads.append(t1)
    
    #[t.start() for t in threads]
    #[t.join() for t in threads]
    #bars.update_bars(pairs, interval)
    

start_proc()


