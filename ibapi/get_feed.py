from wrapper_v4 import IBWrapper, IBclient
from swigibpy import Contract 
import time
import pandas as pd
from time import gmtime, strftime, localtime, sleep
import json
import datetime
from pandas.io.json import json_normalize
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import logging

callback = IBWrapper()
client=IBclient(callback)

def reconnect_ib():
    global callback
    global client
    client.tws.eDisconnect()
    callback = IBWrapper()
    client=IBclient(callback)

def get_bar(symbol, currency):
    return client.get_bar(str(symbol), str(currency))
    
def get_ask(symbol, currency):
    global client
    ask=client.get_IBAsk(str(symbol), str(currency))
    return ask
    
def get_bid(symbol, currency):
    global client
    return client.get_IBBid(symbol, currency)
   
def get_feed(sym, currency, exchange, type, tickerId):
    global tickerid, client, callback
    # Simple contract for GOOG
    contract = Contract()
    contract.symbol = sym
    contract.secType = type
    contract.exchange = exchange
    contract.currency = currency
    ans=client.get_IB_market_data(contract, tickerId)

def get_realtimebar(sym, currency, exchange, type, whatToShow, data, filename, tickerId):
    global client, callback
    # Simple contract for GOOG
    contract = Contract()
    contract.symbol = sym
    contract.secType = type
    contract.exchange = exchange
    contract.currency = currency
    client.get_realtimebar(contract, whatToShow, tickerId, data, filename)

def proc_history(sym, currency, exchange, type, whatToShow, data, filename, tickerId):
    global client, callback
    return client.proc_history(tickerId, sym, type, currency, data)
   

def getDataFromIB( brokerData,endDateTime,data):
    data=client.getDataFromIB(brokerData,endDateTime,data)
    return data

def get_history(date, symbol, currency, exchange, type, whatToShow,data,filename,tickerId, minDataPoints, durationStr, barSizeSetting):
        print "Requesting history for: " + symbol + currency + " ending: " + date
        brokerData = {}
        brokerData =  {'port':7496, 'client_id':101,\
                             'tickerId':tickerId, 'exchange':exchange,'symbol':symbol,\
                             'secType':type,'currency':currency,\
                             'endDateTime':date, 'durationStr':durationStr,\
                             'barSizeSetting':barSizeSetting,\
                             'whatToShow':whatToShow, 'useRTH':0, 'formatDate': 1 \
                              }
                              
         #for date in getHistLoop:
        if data.shape[0] < minDataPoints:
            while data.shape[0] < minDataPoints:      
                data = getDataFromIB(brokerData, date, data)
                data=get_bar(symbol, currency)
                if data.shape[0] > 0:
                    logging.info("Received Date: " + str(data.index[0]) )
                    date = data.index.to_datetime()[0]
                    #eastern=timezone('US/Eastern')
                    #date=date.astimezone(eastern)
                    date=date.strftime("%Y%m%d %H:%M:%S EST")
                    brokerData['endDateTime']=date
                    
                    time.sleep(30)
        else:
            data = getDataFromIB(brokerData, date, data)
            data=get_bar(symbol, currency)
            logging.info("Received Date: " + str(data.index[0]) )
            #update date
            date = data.index.to_datetime()[0]
            #eastern=timezone('US/Eastern')
            #date=date.astimezone(eastern)
            date=date.strftime("%Y%m%d %H:%M:%S EST")
            brokerData['endDateTime']=date
            
            time.sleep(30)
        #set date as last index for next request
    
            
        return data #get_bar(symbol, currency)
        
