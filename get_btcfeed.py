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

logging.basicConfig(filename='/logs/get_btcfeed.log',level=logging.DEBUG)

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
    if exchange == 'bitstampUSD':
        #print 'bitstmp ask' + str(feed[exchange]['ask'])
        return feed[exchange]['ask']
    else:
        return feed[exchange]['price']

def get_btc_bid(ticker, exchange):
    if exchange == 'bitstampUSD':
        #print 'bitstmp bid' + str(feed[exchange]['bid'])
        return feed[exchange]['bid']
    else:
        return feed[exchange]['price']

def get_btcfeed():
    global feed
    global ohlc
    # The IP address or hostname of your reader
    READER_HOSTNAME = 'api.bitcoincharts.com'
    # The TCP port specified in Speedway Connect
    READER_PORT = 27007
    # Define the size of the buffer that is used to receive data.
    BUFFER_SIZE = 4096
     
    # Open a socket connection to the reader
    s = socket.create_connection((READER_HOSTNAME, READER_PORT))
         
    # Set the socket to non-blocking
    #s.setblocking(0)
     
    # Make a file pointer from the socket, so we can read lines
    fs=s.makefile()
     
    # Receive data in an infinite loop
    while 1:
        try:
            line = fs.readline()
            # If data was received, print it
            if (len(line)):
                #print line
                jsondata = json.loads(line)
                if len(jsondata) > 1:
                    dataSet=json_normalize(jsondata).iloc[-1]
                    vol=dataSet['volume']
                    timestamp=dataSet['timestamp']
                    price=dataSet['price']
                    exchange=dataSet['symbol']
                    feedid=dataSet['id']
                    #print "Price: " + str(price)
                    if exchange != 'bitstampUSD':
                        feed[exchange]=dataSet
                    feed_to_ohlc('BTCUSD',exchange, price, timestamp, vol)
        except Exception as e:
            logging.error("get_btcfeed", exc_info=True)
                
    return


def get_ohlc(ticker, exchange):
    global feed
    global ohlc
    global systemdata
    global commissiondata
    if exchange not in hashist:
        get_bthist(ticker, exchange)
        hashist[exchange]=True
        
    ohlc[exchange]=feed_ohlc_to_csv(ticker, exchange)
    return ohlc[exchange]
    
#################################
### Bitstamp Feed             ###
def bitstamp_order_book_callback(data):
    #print "book", data
    try:
        jsondata = json.loads(data)
        dataSet=json_normalize(jsondata).iloc[-1]
        if not feed.has_key('bitstampUSD'):
            feed['bitstampUSD']=dict()
            feed['bitstampUSD']['bid']=-1
            feed['bitstampUSD']['ask']=-1
            
        if 'bids' in dataSet:
            bids=dataSet['bids']
            data=list()
            for bid in bids:
                data.append(bid[0])
            feed['bitstampUSD']['bid']=float(bids[0][0])
            feed_to_ohlc('BTCUSD','bitstampUSD', float(bids[0][0]), str(int(time.time())), 0)
           
        if 'asks' in dataSet:
            asks=dataSet['asks']
            #data=list()
            #for ask in asks:
            #    data.append(ask[0])
            feed['bitstampUSD']['ask']=float(asks[0][0])
            feed_to_ohlc('BTCUSD','bitstampUSD', float(asks[0][0]), str(int(time.time())), 0)
        #print "book", data
    except Except as e:
        logging.error("bitstamp_order_book_callback", exc_info=True)
    
def bitstamp_trade_callback(data):
    try:
        #print "trade", data
        #trade {"price": 408.80000000000001, "amount": 0.076399999999999996, "id": 10832011}
        jsondata = json.loads(data)
        dataSet=json_normalize(jsondata).iloc[-1]
        eastern=timezone('US/Eastern')
        localendDateTime=dt.now(get_localzone())
        localdate=localendDateTime.astimezone(eastern)
        timestamp=int(localdate.strftime("%s"))
        vol=float(dataSet['amount'])
        price=float(dataSet['price'])
        exchange='bitstampUSD'
        feedid=dataSet['id']
        #print exchange + str(price) + ' ' + str(timestamp) + ' ' + str(vol)
        #print "Price: " + str(price)
        #feed[exchange]=dataSet
        feed_to_ohlc('BTCUSD',exchange, float(price), int(timestamp), float(vol))
        #print "trade", data
    except Exception as e:
        logging.error("bitstamp_trade_callback", exc_info=True)
    
    
def bitstamp_connect_handler(data): #this gets called when the Pusher connection is established
    trades_channel = pusher.subscribe("live_trades")
    trades_channel.bind('trade', bitstamp_trade_callback)

    order_book_channel = pusher.subscribe('order_book');
    order_book_channel.bind('data', bitstamp_order_book_callback)

    #orders_channel = pusher.subscribe("live_orders")
    #orders_channel.bind('order_deleted', order_deleted_callback)
    #orders_channel.bind('order_created', order_created_callback)
    #orders_channel.bind('order_changed', order_changed_callback)
    
def get_bitstampfeed():
    global pusher
    pusher = pusherclient.Pusher("de504dc5763aeef9ff52")
    pusher.connection.logger.setLevel(logging.WARNING) #no need for this line if you want everything printed out by the logger
    pusher.connection.bind('pusher:connection_established', bitstamp_connect_handler)
    pusher.connect()
### Bitstamp Feed             ###
#################################
    
def get_signal():
    global feed
    global ohlc
    global systemdata
    global commissiondata
    while True:
      try:
       myfeed=feed.copy()
       for exchange in myfeed:
            
        ticker='BTCUSD'
        if get_btc_ask(ticker, exchange) > 0 and get_btc_bid(ticker, exchange) > 0:
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
                    time.sleep(1)
      except Exception as e:
        #f=open ('./debug/get_btcfeed_cerrors.log','a')
        #f.write(e)
        #f.close()
        logging.error("get_signal", exc_info=True)

#########################
### Paper System Trade###
def get_models(systems):
    v1sList=dict()
    dpsList=dict()
    for i in systems.index:
        system=systems.ix[i]
        if system['ibsym'] == 'BTC':
         
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
    while not finished:
        try:
            (model_pos, dps_model_pos)=get_models(systems)
            symbols=systems['c2sym'].values
            for symbol in symbols:
              system=systems.loc[symbol].copy()
              ticker='BTCUSD'
              exchange=system['ibexch']
              secType=system['ibtype']
                
              if feed.has_key(symbol) \
                  and \
                  get_btc_ask(ticker, exchange) > 0 and get_btc_bid(ticker, exchange) > 0 and \
                  get_timestamp() - int(system['last_trade']) > int(system['trade_freq']):
                     
                model=model_pos
                if system['Version'] == 'v1':
                        model=model_pos
                else:
                        model=dps_model_pos
                
                ask=float(get_btc_ask(ticker, exchange))
                bid=float(get_btc_bid(ticker, exchange))
                
                commissionkey=system['c2sym']+system['ibcur']+system['ibexch']
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

                    print "Processing " + system['Name'] + " Symbol: " + system['ibsym'] + system['ibcur'] + \
                    " Timestamp: " + str(get_timestamp()) + " Last Trade: " + str(system['last_trade']) + " Freq: " +  str(system['trade_freq'])
                    #print "System Name: " + system['Name'] + " Symbol: " + system['ibsym'] + " Currency: " + system['ibcur']
                    #print        " System Algo: " + str(system['System']) 
                    #print        " Ask: " + str(ask)
                    #print        " Bid: " + str(bid)
                    #print        " Commission Pct: " + str(commission_pct*100) + "% Commission Cash: " + str(commission_cash)
                    #print        " Signal: " + str(system_ibpos_qty)
                pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                if ask > 0 and bid > 0:
                    eastern=timezone('US/Eastern')
                    endDateTime=dt.now(get_localzone())
                    date=endDateTime.astimezone(eastern)
                    date=date.strftime("%Y%m%d %H:%M:%S EST")
                    adj_size(model, system['System'],system['Name'],pricefeed,\
                    str(system['c2id']),system['c2api'],system['c2qty'],system['c2sym'],system['c2type'], system['c2submit'], \
                        system['ibqty'],system['c2sym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'], date)
                #time.sleep(1)
                system['last_trade']=get_timestamp()
                systems.loc[symbol]=system
                            
            time.sleep(10)
        except Exception as e:
            #f=open ('./debug/papererrors.log','a')
            #f.write(e)
            #f.close()
            logging.error("something bad happened", exc_info=True)
            
def start_systems():
      global commissiondata
      global threads         
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


def getSystemList():
    systemdata=pd.read_csv('./data/systems/system.csv')
    systemdata=systemdata.reset_index()
    currencyList=dict()
    systemList=dict()

    for i in systemdata.index:
        system=systemdata.ix[i]
        if system['ibsym'] == 'BTC' and system['System'] != 'BTCUSD':
         
          currencyList[system['c2sym']]=1
          
          if systemList.has_key(system['Name']):
              systemList[system['Name']]=systemList[system['Name']].append(system)
          else:
              systemList[system['Name']]=pd.DataFrame()
              systemList[system['Name']]=systemList[system['Name']].append(system)
          
    return systemList            
### Paper System Trade###
#########################       


threads = []
feed_thread = threading.Thread(target=get_btcfeed)
feed_thread.daemon=True
signal_thread = threading.Thread(target=get_signal)
signal_thread.daemon=True
threads.append(feed_thread)
threads.append(signal_thread)

get_bitstampfeed()
feed_thread.start()
start_systems()
get_signal()
 


        
