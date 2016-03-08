import numpy as np
import pandas as pd
import time
import json
from time import gmtime, strftime, time, localtime

from pandas.io.json import json_normalize
from btapi.get_hist_coindesk import  get_hist_coindesk
from btapi.get_hist_blockchain import  get_hist_blockchain
from btapi.raw_to_ohlc import raw_to_ohlc
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

def get_1coin_hist():
    os.system("curl http://api.bitcoincharts.com/v1/csv/1coinUSD.csv.gz > BTCUSD.csv.gz")
    os.system("gunzip BTCUSD.csv.gz")
    raw_to_ohlc('./BTCUSD.csv','./BTCUSD.csv')
    sort_dates('./BTCUSD.csv','./data/btapi/BTCUSD.csv')
    os.remove('./BTCUSD.csv')
    
get_coindesk_hist()
get_blockchain_hist()
get_1coin_hist()
