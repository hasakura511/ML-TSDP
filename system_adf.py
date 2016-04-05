import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from ibapi.get_feed import get_feed, get_realtimebar,getDataFromIB, get_ask as get_ib_ask, get_bid as get_ib_bid
from c2api.place_order import place_order as place_c2order
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
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
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
import os
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual, generate_model_sig, get_model_sig
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
import adfapi.s105 as astrat
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf
import seitoolz.bars as bars


logging.basicConfig(filename='/logs/system_adf.log',level=logging.DEBUG)

pairs=[]
pparams=dict()

if len(sys.argv) > 1 and sys.argv[1] == 's105_EURJPY_CADJPY':
    pairs=[
           ['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EC_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']],
           ['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_EC_CADJPY']]
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CHFJPY']],
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_USDJPY_CHFJPY':
    pairs=[
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EURJPY']],
           ['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_UC_USDJPY']],
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CADJPY']],
           ['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_UC_CHFJPY']]
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_EURJPY_USDJPY':
    pairs=[
           ['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EU_EURJPY']],
           ['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_EU_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CHFJPY']]
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_EURJPY_GBPJPY':
    pairs=[
           ['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EG_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CHFJPY']]
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_EG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_USDJPY_CADJPY':
    pairs=[
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EURJPY']],
           ['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_UCA_USDJPY']],
           ['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_UCA_CADJPY']]
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CHFJPY']]
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_CHFJPY_GBPJPY':
    pairs=[
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CADJPY']],
           ['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CG_CHFJPY']],
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_CG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_EURJPY_CHFJPY':
    pairs=[
           ['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CADJPY']],
           ['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_ECH_CHFJPY']]
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_CG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_CADJPY_CHFJPY':
    pairs=[
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           ['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CC_CADJPY']],
           ['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CC_CHFJPY']]
           #['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_CG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_CADJPY_GBPJPY':
    pairs=[
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           ['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CGB_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CC_CHFJPY']]
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_CGB_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_AUDJPY_GBPJPY':
    pairs=[
           ['./data/from_IB/1 min_AUDJPY.csv', 'AUDJPY', [10,'JPY','IDEALPRO', 's105_AG_AUDJPY']],
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CGB_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CC_CHFJPY']]
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_AG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_NZDJPY_GBPJPY':
    pairs=[
           ['./data/from_IB/1 min_NZDJPY.csv', 'NZDJPY', [10,'JPY','IDEALPRO', 's105_NZG_NZDJPY']],
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           #['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_USDJPY']]
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CGB_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CC_CHFJPY']]
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_NZG_GBPJPY']]
           ]
elif len(sys.argv) > 1 and sys.argv[1] == 's105_USDJPY_GBPJPY':
    pairs=[
           #['./data/from_IB/1 min_NZDJPY.csv', 'NZDJPY', [10,'JPY','IDEALPRO', 's105_NZG_NZDJPY']],
           #['./data/from_IB/1 min_EURJPY.csv', 'EURJPY', [10,'JPY','IDEALPRO', 's105_ECH_EURJPY']],
           ['./data/from_IB/1 min_USDJPY.csv', 'USDJPY', [10,'JPY','IDEALPRO', 's105_UG_USDJPY']],
           #['./data/from_IB/1 min_CADJPY.csv', 'CADJPY', [10,'JPY','IDEALPRO', 's105_CGB_CADJPY']],
           #['./data/from_IB/1 min_CHFJPY.csv', 'CHFJPY', [10,'JPY','IDEALPRO', 's105_CC_CHFJPY']]
           ['./data/from_IB/1 min_GBPJPY.csv', 'GBPJPY', [10,'JPY','IDEALPRO', 's105_UG_GBPJPY']]
           ]
else:
    pairs=[['./data/from_IB/BTCUSD_bitfinexUSD.csv', 'BTCUSD_bitfinexUSD', [10, 'USD', 'bitfinexUSD', 's105_bitfinexUSD']],
           ['./data/from_IB/BTCUSD_bitstampUSD.csv', 'BTCUSD_bitstampUSD', [10, 'USD', 'bitstampUSD', 's105_bitstampUSD']]]

def prep_pair(sym1, sym2, param1, param2):
        global pos
        symPair=sym1+sym2
        if not pos.has_key(symPair):
            pos[symPair]=dict()

        params=dict()
        
        params[sym1]=param1
        params[sym2]=param2
        
        confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
        print "Coint Confidence: " + str(confidence) + "%"
        for i in SST.index:
            try:
                priceHist=SST.ix[i]
                
                timestamp=time.mktime(priceHist['timestamp'].timetuple())
                bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                signals=astrat.procBar(bar1, bar2, pos[symPair], False)
                #proc_signals(signals, params, symPair, timestamp)
                
                
            except Exception as e:
                 logging.error('prep_pair', exc_info=True)
                

def proc_pair(sym1, sym2, param1, param2):
    while 1:
        try:
            proc_onBar(sym1, sym2, param1, param2)
            time.sleep(20)
        except Exception as e:
            logging.error("proc_pair", exc_info=True)
                
def proc_onBar(sym1, sym2, param1, param2):
        logging.info("Processing Bar: " + sym1 + sym2)
                
        symPair=sym1+sym2
        
        params=dict()
        params[sym1]=param1
        params[sym2]=param2
    
        try:
            bardict[sym1]=get_bar(sym1)
            bardict[sym2]=get_bar(sym2)
            timestamp=int(time.time())
            date=datetime.datetime.fromtimestamp(
                timestamp
            ).strftime('%Y-%m-%d %H:%M:00') 
            bardate=time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:00").timetuple())
            
            if not lastDate.has_key(sym1):
                lastDate[sym1]=bardate
            if not lastDate.has_key(sym2):
                lastDate[sym2]=bardate    
                
            if lastDate[sym1] < timestamp and lastDate[sym2] < timestamp:
                lastDate[sym1]=bardate
                lastDate[sym2]=bardate
                timestamp=bardate
                                    
                bar1=astrat.getBar(bardict[sym1], sym1, int(timestamp))
                bar2=astrat.getBar(bardict[sym2], sym2, int(timestamp))
                signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                proc_signals(signals, params, symPair, timestamp)
            
        except Exception as e:
            logging.error("proc_onBar", exc_info=True)
                
def get_entryState():
    global pairs
    
    for [file, sym, param] in pairs:
        try:
            (sysqty, syscur, sysexch, system)=param
            
            signal=get_model_sig(system)
            if len(signal.index) > 0:
                signal=signal.iloc[-1]
                #print signal['comment']
                jsondata = json.loads(signal['comment'])
                entryState=jsondata['Entry']
                exitState=jsondata['Exit']
                symPair=jsondata['symPair']
                logging.info('SymPair: ' + symPair + ' System: ' + system + ' Entry: ' + str(entryState) + ' Exit: ' + str(exitState))
                astrat.updateEntry(symPair, entryState, exitState)
        except Exception as e:
            logging.error("get_entryState", exc_info=True)

def proc_signals(signals, params, symPair, timestamp):
    global pos
    global totalpos
    
    if not pos.has_key(symPair):
            pos[symPair]=dict()
            
    if signals and len(signals) >= 1:
        for signal in signals:
            (barSym, barSig, barCmt)=signal
            
            if pos[symPair].has_key(barSym):
                pos[symPair][barSym]=pos[symPair][barSym] + barSig
            else:
                pos[symPair][barSym]=barSig
                
            if totalpos.has_key(barSym):
                totalpos[barSym]=totalpos[barSym] + barSig
            else:
                totalpos[barSym]=barSig
            
            (sysqty, syscur, sysexch, sysfile)=params[barSym]
            generate_model_sig(sysfile, str(timestamp), totalpos[barSym], 1, barCmt)
           
            if totalpos[barSym] == 0:
                totalpos.pop(barSym, None)
                
            if pos[symPair][barSym] == 0:
                pos[symPair].pop(barSym, None)


def start_bar():
    global SST
    global pairs
    tickers=np.array(pairs,dtype=object)[:,1]
    bars.get_last_bars(tickers, 'Close', onBar)

def onBar(bar, symbols):
    try:
        global SST
        bar=bar.set_index('Date')
        SST = SST.combine_first(bar).sort_index()
        SST = SST.fillna(method='pad')
        logging.info("onBar: " + str(SST.index[-1]))
        for sym in symbols:
            logging.info("onBar: " + sym)
        #Proc
        seen=dict()
        if len(symbols) > 1:
            #for [file1, sym1, param1] in pairs:
            #    for [file2, sym2, param2] in pairs:
            #        if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
            #		 seen[sym1+sym2]=1
            #		 seen[sym2+sym1]=1
            #		 (sym1,sym2,mult1,mult2)=pparams[sym1+sym2]
            #		 proc_onBar(sym1,sym2,mult1,mult2)
            for sym1 in symbols:
                for sym2 in symbols:
                    if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
                        seen[sym1+sym2]=1
                        seen[sym2+sym1]=1
                        (sym1,sym2,mult1,mult2)=pparams[sym1+sym2]
                        proc_onBar(sym1,sym2,mult1,mult2)
    except Exception as e:
        logging.error("onBar", exc_info=True)
    
def get_bar(sym):
    global SST
    return SST.iloc[-1][sym]
    
def start_prep():
    global pairs
    global SST
    seen=dict()
    #Prep
    threads = []
    for [file1, sym1, mult1] in pairs:
        #print "sym: " + sym1
        for [file2, sym2, mult2] in pairs:
            if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
		logging.info("Prepping " + sym1 + sym2)
                seen[sym1+sym2]=1
                seen[sym2+sym1]=1
                
                pparams[sym1+sym2]=[sym1,sym2,mult1,mult2]
                pparams[sym2+sym1]=[sym1,sym2,mult1,mult2]
                
                sig_thread = threading.Thread(target=prep_pair, args=[sym1, sym2, mult1, mult2])
                sig_thread.daemon=True
                threads.append(sig_thread)
    [t.start() for t in threads]
    [t.join() for t in threads]
    #threads=[]
    #seen=dict()
   
    #Proc
    #for [file1, sym1, param1] in pairs:
    #    #print "sym: " + sym1
    #    for [file2, sym2, param2] in pairs:
    #        if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
    #		logging.info("Processing " + sym1 + sym2)
    #            seen[sym1+sym2]=1
    #            seen[sym2+sym1]=1
    #            sig_thread = threading.Thread(target=proc_pair, args=[sym1, sym2, param1, param2])
    #            sig_thread.daemon=True
    #            threads.append(sig_thread)
    #[t.start() for t in threads]
    
    

pos=dict()
totalpos=dict()
bardict=dict()
lastDate=dict()
SST=bars.get_bar_history(pairs, 'Close')
if SST.shape[0] > 3000:
    SST=SST.tail(3000)
get_entryState()
start_prep()
start_bar()




