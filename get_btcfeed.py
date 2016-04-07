import socket   
import select
import sys
import pytz
import datetime
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
import threading
#lock = threading.Lock()


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
    if feed.has_key(exchange):
        if exchange == 'bitstampUSD':
            #print 'bitstmp ask' + str(feed[exchange]['ask'])
            return feed[exchange]['ask']
        else:
            return feed[exchange]['price']
    else:
        return -1

def get_btc_bid(ticker, exchange):
    if feed.has_key(exchange):
        if exchange == 'bitstampUSD':
            #print 'bitstmp bid' + str(feed[exchange]['bid'])
            return feed[exchange]['bid']
        else:
            return feed[exchange]['price']
    else:
        return -1

def get_btcfeed():
    get_bitstampfeed()
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

def get_btc_history():
    global feed
    global ohlc
    global systemdata
    global commissiondata
    while True:
      try:
       myfeed=bars.get_btc_exch_list()
       for exchange in myfeed: 
        ticker='BTCUSD'
        if get_btc_bid(ticker, exchange) > 0 and get_btc_bid(ticker, exchange) > 0:
            get_ohlc(ticker, exchange)
       time.sleep(5)
      except Exception as e:
          logging.error("get_history", exc_info=True)
          
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
    except Exception as e:
        logging.error("bitstamp_order_book_callback", exc_info=True)
    
def bitstamp_trade_callback(data):
    try:
        #print "trade", data
        #trade {"price": 408.80000000000001, "amount": 0.076399999999999996, "id": 10832011}
        jsondata = json.loads(data)
        dataSet=json_normalize(jsondata).iloc[-1]
        
        timestamp=int(time.time())
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
    

def write_feed():
    global feed
    global ohlc
    global systemdata
    global commissiondata
    while 1:
        try:
            eastern=timezone('US/Eastern')
            nowDate=datetime.datetime.now(get_localzone()).astimezone(eastern).strftime('%Y%m%d %H:%M:%S') 
            for exchange in feed.keys():
                bars.bidask_to_csv('BTCUSD_'+exchange, nowDate, get_btc_bid('BTCUSD',exchange), get_btc_ask('BTCUSD',exchange))
                
                feed_ohlc_to_csv('BTCUSD',exchange)
            time.sleep(20)
        except Exception as e:
            logging.error("write_feed", exc_info=True)


threads = []
feed_thread = threading.Thread(target=get_btcfeed)
feed_thread.daemon=True

hist_thread = threading.Thread(target=get_btc_history)
hist_thread.daemon=True
threads.append(hist_thread)
#write_feed_thread.start()
#start_systems()
#get_signal()
#write_feed_thread = threading.Thread(target=write_feed)
#write_feed_thread.daemon=True
#threads.append(write_feed_thread)
#get_bitstampfeed()
threads.append(feed_thread)
feed_thread.start()
hist_thread.start()
write_feed()


        
