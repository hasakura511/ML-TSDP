
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

def start_proc():
    threads = []

    #Currencies
    intervals = ['30m','1h','10m','1 min']
    pairs=bars.get_currencies()
    for interval in intervals:
        t1 = threading.Thread(target=check_bar, args=[pairs, interval])
        t1.daemon=True
        threads.append(t1)
    
    #BTC
    btc=bars.get_btc_list()
    t1 = threading.Thread(target=check_bar, args=[btc, 'choppy', False])
    t1.daemon=True
    threads.append(t1)
    
    [t.start() for t in threads]
    #[t.join() for t in threads]
    while 1:
        time.sleep(3600)

def send_alert(subject, msg, tradingHours=True):
    logging.info(msg)
    print msg
    
    # Alert Mail
    if tradingHours:
        eastern=timezone('US/Eastern')
        d = datetime.datetime.now(get_localzone()).astimezone(eastern)
        print str(d)
        if d.weekday() == 5:
            logging.info("Fortunately Saturday - No Email")
            print "Fortunately Saturday - No Email"
            return
        if d.weekday() == 6 and d.hour < 18:
            logging.info("Fortunately Sunday - No Email")
            print "Fortunately Sunday - No Email"
            return
        if d.weekday() == 4 and d.hour > 17:
            logging.info("Fortunately Friday After Trading Hours - No Email")
            print "Fortunately Friday After Trading Hours - No Email"
            return
    
    try:
        logging.info("Trading Hour - Sending Email Alert")
        print "Trading Hour - Sending Email Alert"
    
        me='catchall@neospace.com'
        you='consulting@neospace.com'
        # Create a text/plain message
        message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
             """ % (me, ", ".join(you), subject, msg)
        
        
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

def check_bar(pairs, interval, tradingHours=True):
    dataPath = './data/from_IB/'
    barPath='./data/bars/'
    while 1:
        try:
            message=''
            for pair in pairs:
                dataFile=dataPath + interval + '_' + pair + '.csv'
                barFile=barPath + interval + '_' + pair + '.csv'
                if interval == '1 min':
                    barFile=barPath + pair + '.csv'
                if interval == 'choppy':
                     dataFile=dataPath + pair + '.csv'
                     barFile=barPath + pair + '.csv'
                if os.path.isfile(dataFile) and os.path.isfile(barFile):
                    
                    data=pd.read_csv(dataFile, index_col='Date')
                    bar=pd.read_csv(barFile, index_col='Date')
                    eastern=timezone('US/Eastern')
                    
                    #timestamp
                    dataDate=parse(data.index[-1]).replace(tzinfo=eastern)
                    barDate=parse(bar.index[-1]).replace(tzinfo=eastern)
                    nowDate=datetime.datetime.now(get_localzone()).astimezone(eastern)
                    dtimestamp = time.mktime(dataDate.timetuple())
                    btimestamp = time.mktime(barDate.timetuple())
                    timestamp=time.mktime(nowDate.timetuple()) + 3600
                    checktime=30
                    if interval == '30m':
                        checktime = 30
                    elif interval == '1h':
                        checktime = 70
                    elif interval == '1 min':
                        checktime = 3
                    elif interval == 'choppy':
                        checktime = 10
                    checktime = checktime * 60
                                        
                    if timestamp - dtimestamp > checktime:
                        message = message + "Feed " + pair + " Interval: " + interval + " Not Updating Since: " + str(data.index[-1]) + '\n'
                        message = message + 'Date:' + str(timestamp) + ' Last Bar:' + str(btimestamp) + ' Last Feed:' + str(dtimestamp) + " Bar Diff " + str(timestamp - btimestamp)+ '\n'
                        message = message + 'Date:' + str(nowDate) + ' Last Bar: ' + str(barDate) + 'co: ' + str(dataDate) + " Data Diff " + str(timestamp - dtimestamp)+ '\n'
                    
                    if timestamp - btimestamp > checktime:
                        message = message + "Bar " + pair + " Interval: " + interval + " Not Updating Since: " + str(data.index[-1]) + '\n'
                        message = message + 'Date:' + str(timestamp) + ' Last Bar:' + str(btimestamp) + ' Last Feed:' + str(dtimestamp) + " Bar Diff " + str(timestamp - btimestamp) + '\n'
                        message = message + 'Date:' + str(nowDate) + ' Last Bar: ' + str(barDate) + 'co: ' + str(dataDate) + " Data Diff " + str(timestamp - dtimestamp)+ '\n'
            if len(message) > 0:
                send_alert(interval + ' Feed Not Updating', message, tradingHours)
            time.sleep(60)
        except Exception as e:
            logging.error("check_bar", exc_info=True)
            
send_alert('Starting System Monitor Services (R) - NSZ Inc.', 'Starting Monitor...')
start_proc()
