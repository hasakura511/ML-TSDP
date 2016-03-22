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

from paper_account import get_account_value, update_account_value
from calc import calc_close_pos, calc_closeVWAP, calc_add_pos, calc_pl

debug=False
        
def get_ib_portfolio(systemname, date):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
    
    account=get_account_value(systemname, 'ib', date)
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
        dataSet=dataSet.reset_index()
        dataSet['symbol']=dataSet['sym'] + dataSet['currency'] 
        dataSet=dataSet.set_index('symbol')
        return (account, dataSet)
    else:

        dataSet=pd.DataFrame({}, columns=['sym','exp','qty','openqty','price','openprice','value','avg_cost','unr_pnl','real_pnl','PurePL','accountid','currency'])
        dataSet['symbol']=dataSet['sym'] + dataSet['currency']        
        dataSet=dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        return (account, dataSet)
   
def get_ib_pos(systemname, symbol, currency, date):
    (account_data, portfolio_data)=get_ib_portfolio(systemname, date)
    portfolio_data=portfolio_data.reset_index()
    portfolio_data['symbol']=portfolio_data['sym'] + portfolio_data['currency']
    sym_cur=symbol + currency
    portfolio_data=portfolio_data.set_index('symbol')
    if sym_cur not in portfolio_data.index.values:
       return 0
    else:
        ib_pos=portfolio_data.loc[sym_cur]
        ib_pos_qty=ib_pos['qty']
        return ib_pos_qty
    
def update_ib_portfolio(systemname, pos, date):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
   
    (account, dataSet)=get_ib_portfolio(systemname, date)
    
    pos=pos.copy()
    symbol = pos['sym'] + pos['currency']
    pos['qty']=pos['openqty']
    pos['price']=pos['openprice']
    pos['symbol']=symbol
    if debug:
        print "Update Portfolio: " + str(symbol)

    if float(pos['qty']) != 0:
        
        if symbol in dataSet.index.values:
            dataSet = dataSet[dataSet.index != symbol]
            dataSet=dataSet.reset_index()
            dataSet=dataSet.append(pos)
            dataSet=dataSet.set_index('symbol')
        else:
            dataSet=dataSet.reset_index()
            dataSet=dataSet.append(pos)
            dataSet=dataSet.set_index('symbol')
            
    else:
        dataSet = dataSet[dataSet.index != symbol]
        
    if debug:
        print "Update Portfolio " + systemname + " Qty: " + str(pos['qty']) + \
                        ' symbol: ' + symbol
    
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'ib', date)
    return (account, dataSet)
# -*- coding: utf-8 -*-

def get_new_ib_pos(systemname, sym, openVWAP, openqty, buy_power, pl, commission, currency, date):
    
    pos=pd.DataFrame([[sym,'',openqty,openVWAP, openVWAP, buy_power, \
                        commission, 0, pl, 'Paper', \
                        currency,openqty,0]], 
                     columns=['sym','exp','qty','price','openprice','value', \
                     'avg_cost','unr_pnl','real_pnl','accountid', \
                     'currency', 'openqty','PurePL']).iloc[-1]
    return pos