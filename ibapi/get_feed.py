from wrapper_v4 import IBWrapper, IBclient
from swigibpy import Contract 
import time
import pandas as pd
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize

def get_feed(sym, currency, exchange, type):
    callback = IBWrapper()
    client=IBclient(callback)


    # Simple contract for GOOG
    contract = Contract()
    contract.symbol = sym
    contract.secType = type
    contract.exchange = exchange
    contract.currency = currency
    
    ans=client.get_IB_market_data(contract, 1000, 111)

