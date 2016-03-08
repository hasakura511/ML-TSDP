import numpy as np
import pandas as pd
import time
import json
from time import gmtime, strftime, time, localtime
from io import StringIO

from pandas.io.json import json_normalize
from btapi.get_hist_coindesk import  get_hist_coindesk
from btapi.get_hist_blockchain import  get_hist_blockchain
from btapi.get_hist_btcharts import  get_hist_btcharts
from btapi.raw_to_ohlc import raw_to_ohlc, raw_to_ohlc_min, raw_to_ohlc_from_csv
from btapi.sort_dates import sort_dates
import os
import sys

def get_blockchain_hist():
    datestr=strftime("%Y%m%d", localtime())
    data=get_hist_blockchain();
    jsondata = json.loads(data)
    dataSet=json_normalize(jsondata['values'])
    dataSet.to_csv('./data/btapi/blockchain' + '_hist.csv', index=False)

def get_coindesk_hist():
    datestr=strftime("%Y%m%d", localtime())
    data=get_hist_coindesk();
    jsondata = json.loads(data)
    dataSet=json_normalize(jsondata['bpi'])
    dataSet.to_csv('./data/btapi/coindesk' + '_hist.csv', index=False)

def get_btcharts_hist(symbol):

    datestr=strftime("%Y%m%d", localtime())
    data=get_hist_btcharts(symbol);
    dataSet = pd.read_csv(StringIO(data))
    dataSet.to_csv('./data/btapi/' + symbol + '_hist.csv', index=False)
    raw_to_ohlc_from_csv('./data/btapi/' + symbol + '_hist.csv','./data/btapi/' + symbol + '_hist.csv')
    sort_dates('./data/btapi/' + symbol + '_hist.csv','./data/btapi/' + symbol + '_hist.csv')
    
def get_1coin_hist():
    os.system("curl http://api.bitcoincharts.com/v1/csv/bitstampUSD.csv.gz > BTCUSD.csv.gz")
    os.system("gunzip BTCUSD.csv.gz")
    raw_to_ohlc_from_csv('./BTCUSD.csv','./BTCUSD.csv')
    sort_dates('./BTCUSD.csv','./data/btapi/BTCUSD.csv')
    os.remove('./BTCUSD.csv')
  
get_btcharts_hist('bitstampUSD')  
get_1coin_hist()
#get_coindesk_hist()
#get_blockchain_hist()
#
