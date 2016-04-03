
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_history, proc_history
from c2api.place_order import place_order as place_c2order
from dateutil.parser import parse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import os
import logging
import re
import threading
import seitoolz.bars as bars
import sys
# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText

logging.basicConfig(filename='/logs/sys_alert.log',level=logging.DEBUG)

intervals = ['30m','1h','10m','1 min']
def start_proc():
    pairs=bars.get_currencies()
    threads = []
    #bars.get_hist_bars(pairs, interval, minDataPoints, exchange, secType)
    #bars.create_bars(pairs, interval)
    for interval in intervals:
        t1 = threading.Thread(target=check_bar, args=[pairs, interval])
        t1.daemon=True
        threads.append(t1)
    
    #t2 = threading.Thread(target=bars.create_bars, args=[pairs, interval])
    #t2.daemon=True
    #threads.append(t2)
    
    [t.start() for t in threads]
    [t.join() for t in threads]

def send_alert(msg):
    logging.info(msg)
    print msg
    try:
        me='catchall@neospace.com'
        you='consulting@neospace.com'
        # Create a text/plain message
        message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
             """ % (me, ", ".join(you), msg, msg)
        
        
        # Send the message via our own SMTP server, but don't include the
        # envelope header.
        server = smtplib.SMTP_SSL('smtp.gmail.com:465')
        server.ehlo()
        #server.starttls()
        server.login('catchall@neospace.com','W0h0h0h0')
        server.sendmail(me, [you], message)
        server.quit()
    except Exception as e:
        logging.error("send_alert", exc_info=True)
    
    
check=dict()

def check_bar(pairs, interval):
    dataPath = './data/from_IB/'
    barPath='./data/bars/'
    while 1:
        try:
            message=''
            for pair in pairs:
                dataFile=dataPath + interval + '_' + pair + '.csv'
                barFile=barPath + interval + '_' + pair + '.csv'
                if os.path.isfile(dataFile) and os.path.isfile(barFile):
                    
                    data=pd.read_csv(dataFile, index_col='Date')
                    bar=pd.read_csv(barFile, index_col='Date')
                    eastern=timezone('US/Eastern')
                    #timestamp
                    dataDate=parse(data.index[-1]).replace(tzinfo=eastern)
                    barDate=parse(bar.index[-1]).replace(tzinfo=eastern)
                    dtimestamp = time.mktime(dataDate.timetuple())
                    btimestamp = time.mktime(barDate.timetuple())
                    timestamp=int(time.time())
                    checktime=30
                    if interval == '30m':
                        checktime = 30
                    elif interval == '1h':
                        checktime = 70
                    elif interval == '1 min':
                        checktime = 3
                    
                    if timestamp - dtimestamp > checktime:
                        message = message + "Feed " + pair + " Interval: " + interval + " Not Updating Since: " + str(data.index[-1]) + '\n'
                    if timestamp - btimestamp > checktime:
                        message = message + "Bar " + pair + " Interval: " + interval + " Not Updating Since: " + str(data.index[-1]) + '\n'
            if len(message) > 0:
                send_alert(message)
            time.sleep(60)
        except Exception as e:
            logging.error("check_bar", exc_info=True)
            
send_alert('Starting System Monitor Services (R) - NSZ Inc.')
start_proc()
