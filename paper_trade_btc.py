import socket   
import select
import sys
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone

import json
import time
from pandas.io.json import json_normalize
import pandas as pd
import threading
from btapi.get_signal import get_v1signal
from btapi.get_hist_btcharts import get_bthist
from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
from seitoolz.paper import adj_size
import sys
import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
import websocket
import datetime
from seitoolz.signal import get_dps_model_pos, get_model_pos
import seitoolz.bars as bars

logging.basicConfig(filename='/logs/paper_trade_btc.log',level=logging.DEBUG)

debug=True

feed={}
ohlc={}
hashist={}
model=pd.DataFrame()

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')


def get_btc_ask(ticker, exchange):
    data=bars.bidask_from_csv(ticker + '_' + exchange).iloc[-1]
    return data['Ask']

def get_btc_bid(ticker, exchange):
    data=bars.bidask_from_csv(ticker + '_' +  exchange).iloc[-1]
    return data['Bid']
    
def get_btc_bidask(ticker, exchange):
    data=bars.bidask_from_csv(ticker + '_' +  exchange).iloc[-1]
    return (data['Bid'],data['Ask'])
    
def get_ohlc(ticker, exchange):
    ohlc[exchange]=bars.feed_ohlc_from_csv(ticker + '_' + exchange)
    return ohlc[exchange]
    
#########################
### Paper System Trade###
def get_models(systems):
    v1sList=dict()
    dpsList=dict()
    for i in systems.index:
        system=systems.ix[i]
        if system['ibtype'] == 'BITCOIN':
         
          if system['Version'] == 'v1':
              v1sList[system['System']]=1
          else:
              dpsList[system['System']]=1
          
    model_pos=get_model_pos(v1sList.keys())
    dps_model_pos=get_dps_model_pos(dpsList.keys())    
    return (model_pos, dps_model_pos)
    
def get_timestamp():
	timestamp = int(time.time())
	return timestamp
        
def start_trade(systems, commissiondata): 
        global debug
        if debug:
           print "Starting " + str(systems.iloc[0]['Name'])
           logging.info("Starting " + str(systems.iloc[0]['Name']))
        finished=False
        #while not finished:
        try:
            (model_pos, dps_model_pos)=get_models(systems)
            symbols=systems['c2sym'].values
            for symbol in symbols:
              system=systems.loc[symbol].copy()
              ticker='BTCUSD'
              exchange=system['ibexch']
              secType=system['ibtype']
              (bid,ask)=get_btc_bidask(ticker, exchange)
              feed=bars.get_btc_exch_list()
              if symbol in feed \
                  and \
                  ask > 0 and bid > 0 and \
                  get_timestamp() - int(system['last_trade']) > int(system['trade_freq']):
                     
                model=model_pos
                if system['Version'] == 'v1':
                        model=model_pos
                else:
                        model=dps_model_pos
                
                ask=float(get_btc_ask(ticker, exchange))
                bid=float(get_btc_bid(ticker, exchange))
                
                commissionkey=system['c2sym']+system['ibcur']+system['ibexch']
                commission_pct=0.0025
                commission_cash=0
                if commissionkey in commissiondata.index:
                    commission=commissiondata.loc[commissionkey]
                    commission_pct=float(commission['Pct'])
                    commission_cash=float(commission['Cash'])
                    
                if debug:
                    system_pos=model.loc[system['System']]
                    system_c2pos_qty=round(system_pos['action']) * system['c2qty']
                    system_ibpos_qty=round(system_pos['action']) * system['ibqty']

                    print "Processing " + system['Name'] + " Symbol: " + system['ibsym'] + system['ibcur'] + \
                    " Timestamp: " + str(get_timestamp()) + " Last Trade: " + str(system['last_trade']) + " Freq: " +  str(system['trade_freq'])
                    print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
                    print        " System Algo: " + str(system['System']) 
                    print        " Ask: " + str(ask)
                    print        " Bid: " + str(bid)
                    print        " Commission Pct: " + str(commission_pct*100) + "% Commission Cash: " + str(commission_cash)
                    print        " Signal: " + str(system_ibpos_qty)
                pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
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
                            
            #time.sleep(10)
        except Exception as e:
            #f=open ('./debug/papererrors.log','a')
            #f.write(e)
            #f.close()
            logging.error("something bad happened", exc_info=True)
            
def start_systems():
      global commissiondata
      threads=list()         
      systemList=getSystemList()
      
      for systemname in systemList.keys():
           systems=systemList[systemname]
           systems['last_trade']=0
           systems['key']=systems['c2sym']
           systems=systems.set_index('key')
           sig_thread = threading.Thread(target=start_trade, args=[systems,commissiondata])
           sig_thread.daemon=True
           threads.append(sig_thread)
           sig_thread.start()
      [t.join() for t in threads]
      #while 1:
      #   time.sleep(1000)

def getSystemList():
    systemdata=pd.read_csv('./data/systems/system.csv')
    systemdata=systemdata.reset_index()
    currencyList=dict()
    systemList=dict()

    for i in systemdata.index:
        system=systemdata.ix[i]
        if system['ibtype'] == 'BITCOIN' and system['System'] != 'BTCUSD':
         
          currencyList[system['c2sym']]=1
          
          if systemList.has_key(system['Name']):
              systemList[system['Name']]=systemList[system['Name']].append(system)
          else:
              systemList[system['Name']]=pd.DataFrame()
              systemList[system['Name']]=systemList[system['Name']].append(system)
          
    return systemList            
### Paper System Trade###
#########################       
start_systems()
 


        
