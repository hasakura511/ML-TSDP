import sys
sys.path.append('..')
import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
import websocket
from websocket import WebSocket as WebSocketApp
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

logging.basicConfig()


feed={}
ohlc={}
hashist={}
model=pd.DataFrame()



def order_deleted_callback(data):
    print "delete", data

def order_created_callback(data):
    print "create", data

def order_changed_callback(data):
    print "changes", data
    
def bitstamp_order_book_callback(data):
    #print "book", data
    mydata=data
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
        feed['bitstampUSD']['bid']=float(max(data))
    if 'asks' in dataSet:
        asks=dataSet['asks']
        data=list()
        for ask in asks:
            data.append(ask[0])
        feed['bitstampUSD']['ask']=float(min(data))    
    #print "book", mydata
    
def bitstamp_trade_callback(data):
    print "trade", data
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
    print "trade", data
    
    
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
    
    while True:  #run until ctrl+c interrupts
        time.sleep(1)

get_bitstampfeed()
