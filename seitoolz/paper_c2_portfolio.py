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
from paper_c2_trades import get_c2_trades

debug=False
debug2=False

def get_c2_portfolio(systemname, date):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    
    account=get_account_value(systemname, 'c2', date)
    
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
        return (account, dataSet)
        
    else:
        dataSet=pd.DataFrame({},columns=['symbol','open_or_closed','long_or_short','quant_opened', 'quant_closed', \
                     'opening_price_VWAP', 'closing_price_VWAP', 'PL', 'PurePL', 'commission', \
                     'trade_id','closeVWAP_timestamp','closedWhen','closedWhenUnixTimeStamp', \
                     'expir','instrument',\
                     'markToMarket_time','openVWAP_timestamp','openedWhen','qty'])
        dataSet = dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        return (account, dataSet)

    
def get_c2_pos(systemname, c2sym, date):
    
    (account_data, portfolio_data)=get_c2_portfolio(systemname, date)
    portfolio_data=portfolio_data.reset_index()
    sym_cur=c2sym
    portfolio_data=portfolio_data.set_index('symbol')
    if sym_cur not in portfolio_data.index.values:
       return 0
    else:
        c2_pos=portfolio_data.loc[sym_cur]
        c2_pos_qty=float(c2_pos['quant_opened']) - float(c2_pos['quant_closed'])
        c2_pos_side=str(c2_pos['long_or_short'])
        if c2_pos_side == 'short':
            c2_pos_qty=-abs(c2_pos_qty)
        return c2_pos_qty

def get_new_c2_pos(systemname, sym, side, openVWAP, openqty, pl, commission, ptValue, date):
    trade_id=0
    trades=get_c2_trades(systemname, date)
    if len(trades.index.values) > 0:
        trade_id=int(max(trades.index.values))
    trade_id=int(trade_id) + 1
    
    if debug:
        print "Trade ID:" + str(trade_id)
            
    pos=pd.DataFrame([[sym, 'open',side, abs(openqty), 0, \
                        openVWAP, 0, 0, 0, commission, \
                        trade_id, '','','', \
                        '',sym, \
                        '',date,date,openqty]]
            ,columns=['symbol','open_or_closed','long_or_short','quant_opened', 'quant_closed', \
                     'opening_price_VWAP', 'closing_price_VWAP', 'PL', 'PurePL', 'commission', \
                     'trade_id','closeVWAP_timestamp','closedWhen','closedWhenUnixTimeStamp', \
                     'expir','instrument',\
                     'markToMarket_time','openVWAP_timestamp','openedWhen','qty']).iloc[-1]
    return pos


def update_c2_portfolio(systemname, pos, tradepl, purepl, buy_power, date):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    (account, dataSet)=get_c2_portfolio(systemname, date)

    pos=pos.copy()
    symbol=pos['symbol']
    
    pos['balance']=account['balance'] + tradepl
    pos['purebalance']=account['purebalance'] + purepl
    pos['margin_available']=account['buy_power'] + buy_power
    pos['PurePL']=float(pos['PurePL']) + purepl
    pos['PL']=float(pos['PL']) + float(tradepl)
    
    if debug2:
         print 'C2 Symbol: ' + pos['symbol'] + ' PurePL: ' + str(pos['PurePL']) + ' From Portfolio'
         
    if debug:
        print "Update Portfolio: " + str(symbol)
        
    if abs(pos['quant_opened']) > abs(pos['quant_closed']):
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
        print "Update Portfolio " + systemname + " " + pos['long_or_short'] + \
                        ' symbol: ' + pos['symbol'] + ' open_or_closed ' + pos['open_or_closed'] + \
                        ' opened: ' + str(pos['quant_opened']) + ' closed: ' + str(pos['quant_closed'])
    dataSet.to_csv(filename)
    account=get_account_value(systemname, 'c2', date)
    return (account, dataSet)