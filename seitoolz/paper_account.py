import numpy as np
import pandas as pd
import time
import os.path

import json
from pandas.io.json import json_normalize
from seitoolz.signal import get_model_pos
from time import gmtime, strftime, localtime, sleep
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone


debug=False

def get_account_value(systemname, broker, date):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['Date'])
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
    else:
        dataSet=make_new_account(systemname, broker, date)
                
    account=pd.DataFrame([[date, dataSet.iloc[-1]['accountid'],
                            dataSet.iloc[-1]['balance'],dataSet.iloc[-1]['purebalance'],
                            dataSet.iloc[-1]['buy_power'],
                            dataSet.iloc[-1]['unr_pnl'], dataSet.iloc[-1]['real_pnl'], 
                            dataSet.iloc[-1]['PurePL'],
                            dataSet.iloc[-1]['currency']]], 
        columns=['Date','accountid','balance','purebalance','buy_power','unr_pnl','real_pnl','PurePL','currency']).iloc[-1]
        
    return account

def update_account_value(systemname, broker, account):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['Date'])
    else:
        dataSet=make_new_account(systemname, broker,account['Date'])

    dataSet=dataSet.reset_index()
    dataSet=dataSet.append(account)
    dataSet=dataSet.set_index('Date')
    dataSet.to_csv(filename)
    #print 'Account Update: ' + broker + ' Balance: ' + str(account['balance']) + ' PurePNL:' + str(account['PurePL'])
    return account

def make_new_account(systemname, broker, date):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    dataSet=pd.DataFrame([[date, 'paperUSD',20000,20000,400000,0,0,0,'USD']], columns=['Date','accountid','balance','purebalance','buy_power','unr_pnl','real_pnl','PurePL','currency'])
    dataSet=dataSet.set_index('Date')
    dataSet.to_csv(filename)
    return dataSet