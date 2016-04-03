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
from seitoolz.signal import get_dps_model_pos, get_model_pos
import seitoolz.bars as bars
import datetime
logging.basicConfig(filename='/logs/paper_trade_btc_v1.log',level=logging.DEBUG)

debug=False

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
    data=bars.bidask_from_csv(ticker, exchange).iloc[-1]
    return data['Ask']

def get_btc_bid(ticker, exchange):
    data=bars.bidask_from_csv(ticker, exchange).iloc[-1]
    return data['Bid']
    
def get_btc_bidask(ticker, exchange):
    data=bars.bidask_from_csv(ticker, exchange).iloc[-1]
    return (data['Bid'],data['Ask'])
    
def get_ohlc(ticker, exchange):
    ohlc[exchange]=bars.feed_ohlc_from_csv(ticker, exchange)
    return ohlc[exchange]

### Bitstamp Feed             ###
#################################
    
def trade_v1():
    global feed
    global ohlc
    global systemdata
    global commissiondata
    while True:
      try:
       myfeed=bars.get_btc_exch_list()
       for exchange in myfeed:
            
        ticker='BTCUSD'
        (bid,ask)=get_btc_bidask(ticker, exchange)
        if ask > 0 and bid > 0:
         data=get_ohlc(ticker, exchange)
         if data.shape[0] > 2000:
            #model=get_v1signal(data.tail(2000), ticker, exchange)
            model=get_v1signal(data, ticker, exchange)
            for i in systemdata.index:
                system=systemdata.ix[i].copy()
                if system['System'] == 'BTCUSD':
                    system['Name']=system['Name'] + '_' + exchange
                    system['System'] = ticker + '_' + exchange
                    
                  
                    ask=float(get_btc_ask(ticker, exchange))
                    bid=float(get_btc_bid(ticker, exchange))
                    secType=system['ibtype']
                    
                    
                    commissionkey=system['ibsym']+system['ibcur']+system['ibexch']
                    commission_pct=0.0025
                    commission_cash=0
                    if commissionkey in commissiondata.index:
                        commission=commissiondata.loc[commissionkey]
                        commission_pct=float(commission['Pct'])
                        commission_cash=float(commission['Cash'])
                   
                    system_pos=model.loc[system['System']]
                    system_c2pos_qty=round(system_pos['action']) * system['c2qty']
                    system_ibpos_qty=round(system_pos['action']) * system['ibqty']
                    if debug:
                        print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
                        print        " System Algo: " + str(system['System']) 
                        print        " Ask: " + str(ask)
                        print        " Bid: " + str(bid)
                        print        " Commission Pct: " + str(commission_pct*100) + "% Commission Cash: " + str(commission_cash)
                        print        " Signal: " + str(system_ibpos_qty)
                    pricefeed=pd.DataFrame([[ask, bid, 10000, 10000, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                    if ask > 0 and bid > 0:
                        eastern=timezone('US/Eastern')
                        endDateTime=dt.now(get_localzone())
                        date=endDateTime.astimezone(eastern)
                        date=date.strftime("%Y%m%d %H:%M:%S EST")

			#print ' Qty: ' + str(system['ibqty']) + ' c2 ' + str(system['c2qty'])
                        adj_size(model, system['System'],system['Name'],pricefeed,\
                        str(system['c2id']),system['c2api'],float(system['c2qty']),system['c2sym'],system['c2type'], system['c2submit'], \
                            float(system['ibqty']),system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'], date)
                    time.sleep(10)
      except Exception as e:
        #f=open ('./debug/get_btcfeed_cerrors.log','a')
        #f.write(e)
        #f.close()
        logging.error("get_signal", exc_info=True)

trade_v1()


        
