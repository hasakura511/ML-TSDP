import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
import ibapi.get_feed as feed
from c2api.place_order import place_order as place_c2order
import threading
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
from swigibpy import EPosixClientSocket, ExecutionFilter, CommissionReport, Execution, Contract
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.paper import adj_size
import seitoolz.bars as bars
from time import gmtime, strftime, localtime, sleep
import logging

logging.basicConfig(filename='/logs/paper_trade.log',level=logging.DEBUG)

currencyList=dict()
systemList=dict()

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
for i in systemdata.index:
      system=systemdata.ix[i]
      currencyList[system['c2sym']]=1
      if systemList.has_key(system['Name']):
          systemList[system['Name']]=systemList[system['Name']].append(system)
      else:
          systemList[system['Name']]=pd.DataFrame()
          systemList[system['Name']]=systemList[system['Name']].append(system)
          
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

debug=False
signalPath = './data/signals/'
dataPath = './data/from_IB/'

if len(sys.argv) > 1 and sys.argv[1] == '1':
    debug=True

def get_timestamp():
	timestamp = int(time.time())
	return timestamp
    
def get_models(systems):
    dpsList=dict()
    for i in systems.index:
        system=systems.ix[i]
        dpsList[system['System']]=1
    dps_model_pos=get_dps_model_pos(dpsList.keys())    
    return dps_model_pos
    
def start_trade(systems, commissiondata): 
        global debug
        if debug:
           print "Starting " + str(systems.iloc[0]['Name'])
           logging.info("Starting " + str(systems.iloc[0]['Name']))
        #finished=False
        #while not finished:
        try:
            model_pos=get_models(systems)
            symbols=systems['c2sym'].values
            for symbol in symbols:
              system=systems.loc[symbol].copy()
              symbol=system['ibsym']
              if system['ibtype'] == 'CASH':
                    symbol = str(system['ibsym']) + str(system['ibcur'])
              
              feed_dict=bars.get_bidask_list()
              if symbol in feed_dict:
                #and get_timestamp() - int(system['last_trade']) > int(system['trade_freq']):
                model=model_pos
                ask=float(bars.get_ask(symbol))
                bid=float(bars.get_bid(symbol))
                exchange=system['ibexch']
                secType=system['ibtype']
                
                commissionkey=system['ibsym']+system['ibcur']+system['ibexch']
                commission_pct=0.00002
                commission_cash=2
                if commissionkey in commissiondata.index:
                    commission=commissiondata.loc[commissionkey]
                    commission_pct=float(commission['Pct'])
                    commission_cash=float(commission['Cash'])
                if debug:
                    system_pos=model.loc[system['System']]
                    system_c2pos_qty=round(system_pos['action']) * system['c2qty']
                    system_ibpos_qty=round(system_pos['action']) * system['ibqty']

                    print "Processing " + system['Name'] + " Symbol: " + symbol + \
                    " Timestamp: " + str(get_timestamp()) + " Last Trade: " + str(system['last_trade']) + " Freq: " +  str(system['trade_freq'])
                    print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
                    print        " System Algo: " + str(system['System']) 
                    print        " Ask: " + str(ask)
                    print        " Bid: " + str(bid)
                    print        " Commission Pct: " + str(commission_pct*100) + "% Commission Cash: " + str(commission_cash)
                    print        " IB Signal: " + str(system_ibpos_qty)
                    print        " C2 Signal: " + str(system_c2pos_qty)
                pricefeed=pd.DataFrame([[ask, bid, float(system['c2mult']), float(system['ibmult']), exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                if ask > 0 and bid > 0:
                    eastern=timezone('US/Eastern')
                    endDateTime=dt.now(get_localzone())
                    date=endDateTime.astimezone(eastern)
                    date=date.strftime("%Y%m%d %H:%M:%S EST")
                    adj_size(model, system['System'],system['Name'],pricefeed,\
                    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
                        system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'], date)
                #time.sleep(1)
                system['last_trade']=get_timestamp()
                systems.loc[symbol]=system
                            
            #time.sleep(30)
        except Exception as e:
            logging.error("something bad happened", exc_info=True)

threads = []
def start_systems(systemList):
      global commissiondata
      global threads         
      for systemname in systemList.keys():
           systems=systemList[systemname]
           systems['last_trade']=0
           systems['key']=systems['c2sym']
           systems=systems.set_index('key')
           sig_thread = threading.Thread(target=start_trade, args=[systems,commissiondata])
           sig_thread.daemon=True
           threads.append(sig_thread)
           sig_thread.start()
      #while 1:
      #    time.sleep(100)
      [t.join() for t in threads]
         
start_systems(systemList)
    

