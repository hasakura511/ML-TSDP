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

def adj_size(model_pos, system, system_name, pricefeed, c2systemid, c2apikey, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit, date):
    system_pos=model_pos.loc[system]
   
    c2submit=True
    ibsubmit=True
    if c2submit:
        c2_pos=get_c2_pos(system_name, c2sym, date)
        c2_pos=c2_pos.loc[c2sym]
        c2_pos_qty=float(c2_pos['quant_opened']) - float(c2_pos['quant_closed'])
        c2_pos_side=str(c2_pos['long_or_short'])
        if c2_pos_side == 'short':
            c2_pos_qty=-c2_pos_qty
            
        system_c2pos_qty=round(system_pos['action']) * c2quant
        
        if system_c2pos_qty != c2_pos_qty:
            if debug:
                print "========"
                print "system: " + system_name + " symbol: " + c2_pos['symbol']
                print "system_c2_pos: " + str(system_c2pos_qty)
                print "c2_pos: " + str(c2_pos_qty)
                print "========"
            
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                if debug:
                    print 'BTC: ' + str(qty)
                place_order(system_name, 'BTC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                                
                c2quant = c2quant - qty
            
            if c2quant > 0:
                if debug:
                    print 'BTO: ' + str(c2quant)
                place_order(system_name, 'BTO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                if debug:
                    print 'STC: ' + str(qty)
                place_order(system_name, 'STC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
                c2quant = c2quant - qty

            if c2quant > 0:
                if debug:
                    print 'STO: ' + str(c2quant)
                place_order(system_name, 'STO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
    if ibsubmit:
        paper_pos=get_ib_pos(system_name, ibsym,ibcurrency, date)
        symbol=ibsym+ibcurrency
        ib_pos=paper_pos.loc[symbol]
        ib_pos_qty=ib_pos['qty']
        system_ibpos_qty=round(system_pos['action']) * ibquant
        
        if system_ibpos_qty != ib_pos_qty:
            if debug:
                print "========"
                print "system: " + system_name + " symbol: " + ib_pos['symbol']
                print "system_ib_pos: " + str(system_ibpos_qty)
                print "ib_pos: " + str(ib_pos_qty)
                print "========"
        
        if system_ibpos_qty > ib_pos_qty:
            ibquant=float(system_ibpos_qty - ib_pos_qty)
            if debug:
                print 'BUY: ' + str(ibquant)
            place_order(system_name, 'BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed,date);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=float(ib_pos_qty - system_ibpos_qty)
            if debug:
                print 'SELL: ' + str(ibquant)
            place_order(system_name, 'SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed,date)

def test(sym, systemname, date):
    (account, portfolio)=get_c2_portfolio(systemname, date)
    if sym in portfolio['symbol'].values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[portfolio['symbol']==sym].reset_index().iloc[-1]
            if debug:
                print pos['trade_id']

def place_order(systemname, action, quant, sym, type, currency, exch, broker, pricefeed, date):
    if debug:
        print "Place Order " + action + " " + str(quant) + " " + sym + " " + currency + " " + broker 
    pricefeed=pricefeed.iloc[-1]
    
    if broker == 'c2':
        (account, portfolio)=get_c2_portfolio(systemname, date)
    
            
        if sym in portfolio.index.values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[sym].copy()
            pos['symbol']=sym
            side=str(pos['long_or_short'])
            openVWAP=float(pos['opening_price_VWAP'])
            closeVWAP=float(pos['closing_price_VWAP'])
            openqty=float(pos['quant_opened'])
            closedqty=float(pos['quant_closed'])
            if side == 'short':
                openqty = -abs(openqty)
            elif side == 'long':
                closedqty = -abs(closedqty)
            
            openVWAP_timestamp=str(pos['openVWAP_timestamp'])
            closeVWAP_timestamp=str(pos['closeVWAP_timestamp'])
            pl=float(pos['PL'])
            tradepl=float(pos['PL'])
            ptValue=float(pos['ptValue'])
            
            if debug:
                print 'Symbol: ' + pos['symbol'] + ' Action: ' + action +' Side: ' + side
                
            if (action == 'BTO' and side=='long') or (action == 'STO' and side=='short'):
                price=pricefeed['Ask']
                if action == 'STO':
                    price=pricefeed['Bid']
                    
                (openVWAP, openqty, commission, buy_power, purepl, ptValue, side)=calc_add_pos(openVWAP,openqty,
                    price, quant, 
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, 
                    pricefeed['C2Mult'])
                pl = pl + purepl           
                tradepl=purepl 
                
                pos['openVWAP_timestamp']=date
                pos['opening_price_VWAP']=openVWAP
                pos['quant_opened']=abs(openqty)
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission
                pos['long_or_short']=side
                
                if debug:
                    print "++++++++++++++++++++"
                    print "Trade ID: " + str(pos['trade_id'])
                    print "C2 " + action + ' ' + str(sym) + "@" + str(price) + "[" + str(quant) + "]"
                    print " Total Opened: " + str(openVWAP) + "[" + str(openqty) + "]" 
                    print " Total Closed: " + str(closeVWAP) + "[" + str(closedqty) + "]" 
                    print "Trade PL: " + str(purepl) + " Total PL: " + str(pl) + " (Commission: " + str(commission) + ")"
                    print "++++++++++++++++++++"
                update_c2_trades(systemname, pos, tradepl, pl, buy_power, date)
                update_c2_portfolio(systemname, pos, date)

                
            if (action == 'BTC' and side=='short') or (action == 'STC' and side=='long'):
                price=pricefeed['Ask']
                if action == 'STC':
                    price=pricefeed['Bid']
                    quant=-abs(quant)
                    
                (closeVWAP, closedqty)=calc_closeVWAP(closeVWAP, closedqty, price, quant)
                (openVWAP, openqty, commission, buy_power, purepl, ptValue,side) =           \
                    calc_close_pos(openVWAP,openqty, price, quant,                      \
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, \
                    pricefeed['C2Mult'])
                pl = pl + purepl
                tradepl=purepl
                
                pos['openVWAP_timestamp']=date
                pos['opening_price_VWAP']=openVWAP
                pos['quant_opened']=abs(openqty)
                pos['closeVWAP_timestamp']=date
                pos['closedWhen']=date
                pos['closedWhenUnixTimeStamp']=date
                pos['closing_price_VWAP']=closeVWAP
                pos['quant_closed']=abs(closedqty)
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission
                pos['long_or_short']=side
                
                if abs(closedqty) == abs(openqty):
                    pos['open_or_closed']='closed'
                
                if debug:
                    print "--------------------"
                    print "Trade ID: " + str(pos['trade_id'])
                    print "C2 " + action + ' ' + str(sym) + "@" + str(price) + "[" + str(quant) + "]"
                    print " Total Opened: " + str(openVWAP) + "[" + str(openqty) + "]" 
                    print " Total Closed: " + str(closeVWAP) + "[" + str(closedqty) + "]" 
                    print "Trade PL: " + str(purepl) + " Total PL: " + str(pl) + " (Commission: " + str(commission) + ")"
                    print "--------------------"
                    
                update_c2_trades(systemname, pos, tradepl, pl, buy_power, date)
                update_c2_portfolio(systemname, pos, date)
                
        elif action == 'STO' or action == 'BTO':
            side='long'
            price=-1
            if action == 'STO':
                side='short'
                price=pricefeed['Bid']
                quant=-abs(quant)
                
            if action == 'BTO':
                side='long'
                price=pricefeed['Ask']
                
            trade_id=0
            trades=get_c2_trades(systemname, date)
            if len(trades.index.values) > 0:
                trade_id=int(max(trades.index.values))
            trade_id=int(trade_id) + 1
            
            if debug:
                print "Trade ID:" + str(trade_id)
            
            (openVWAP, openqty, commission, buy_power,purepl, ptValue, side)=calc_add_pos(0,0,
                    price, quant, 
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, 
                    pricefeed['C2Mult'])
                    
            pl=purepl
            tradepl=purepl
            
            pos=pd.DataFrame([[trade_id, pl, '','', \
                                '',0,'',sym,side, \
                                date,date,'open',date,openVWAP, \
                                ptValue,'',0,abs(openqty),'',sym,systemname, \
                                commission]] \
            ,columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
            'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
            'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
            'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description', \
            'commission']).iloc[-1]
            update_c2_trades(systemname, pos, tradepl, pl, buy_power, date)
            update_c2_portfolio(systemname, pos, date)
        else:
            print action + ' ' + side + ' ILLEGAL OP '
    if broker == 'ib':
        (account, portfolio)=get_ib_portfolio(systemname, date)
        
        symbol=sym+currency
        if symbol in portfolio.index.values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[symbol].copy()
            pos['symbol']=symbol
            openVWAP=float(pos['openprice'])
            openqty=float(pos['openqty'])
            side='long'
            if openqty < 0:
                side='short'
                
            pl=0
            ptValue=float(pricefeed['IBMult'])
            
            if debug:
                print 'Symbol: ' + pos['sym'] + pos['currency'] + ' Action: ' + action +' Side: ' + side
                
            if (action == 'BUY' and side == 'long') or (action == 'SELL' and side=='short'):
                price=pricefeed['Ask']
                quant=abs(quant)
                
                if (action == 'SELL'):
                    price=pricefeed['Bid']
                    quant=-abs(quant)
                
                (openVWAP, openqty, commission, buy_power,purepl, ptValue,side)=calc_add_pos(openVWAP,openqty, \
                    price, quant,  \
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, \
                    pricefeed['IBMult'])
                    
                pl = pl + purepl 
                       
                pos['openprice']=openVWAP
                pos['openqty']=openqty
                pos['price']=price
                pos['qty']=quant
                pos['value']=buy_power
                pos['avg_cost']=commission
                pos['real_pnl']=pl
                pos['unr_pnl']=0
                
                if debug:
                    print "IB " + action + ' ' + str(symbol) + "@" + str(price) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                    print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"   
                    print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"
                
                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos, date)
                
            if (action == 'BUY' and side=='short') or (action == 'SELL' and side=='long'):
                price=pricefeed['Ask']
                
                if (action == 'SELL'):
                    price=pricefeed['Bid']
                    quant=-abs(quant)
 
                (openVWAP, openqty, commission, buy_power, purepl, ptValue,side) =           \
                    calc_close_pos(openVWAP,openqty, price, quant,                      \
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, \
                    pricefeed['IBMult'])
                pl = purepl

                pos['openprice']=openVWAP
                pos['openqty']=openqty
                pos['price']=price
                pos['qty']=quant
                pos['value']=buy_power
                pos['real_pnl']=pl
                pos['avg_cost']=commission
                pos['unr_pnl']=0
                
                if debug:
                    print "IB " + action + " " + str(symbol) + "@" + str(price) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                    print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                    print "Trade PL: " + str(purepl)
                    print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"

                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos, date)
                    
            
        else:
            side='long'
            price=-1
            if action == 'SELL':
                side='short'
                price=pricefeed['Bid']
                quant=-abs(quant)
                
            if action == 'BUY':
                side='long'
                price=pricefeed['Ask']
                quant=abs(quant)
            
            (openVWAP, openqty, commission, buy_power,pl, ptValue,side)=calc_add_pos(0,0,price,quant,
                    pricefeed['Commission_Pct'],pricefeed['Commission_Cash'], currency, 
                    pricefeed['IBMult'])
             
            pos=pd.DataFrame([[sym,'',openqty,openVWAP,openVWAP, buy_power, \
                                    commission, 0, pl, 'Paper', \
                                    currency,openqty]], 
                                 columns=['sym','exp','qty','price','openprice','value', \
                                 'avg_cost','unr_pnl','real_pnl','accountid', \
                                 'currency', 'openqty']).iloc[-1]
            
            update_ib_trades(systemname, pos, pl, buy_power, exch, date)
            update_ib_portfolio(systemname, pos, date)   


def calc_close_pos(openAt, openqty, closeAt, closeqty, comPct, comCash, currency, mult):
    purepl=0
    
    commission=abs(comPct * closeAt * closeqty * mult)     
    qty=abs(abs(openqty) - abs(closeqty))
    side='long'
    if openqty < 0:         
        side='short'
        qty = -qty
    
    if abs(closeqty) > abs(openqty):
        closeqty=abs(openqty)
        qty = -qty
        
    (purepl, value)=calc_pl(openAt, closeAt, closeqty, mult, side)
    
    if currency != 'USD':
        purepl=purepl * get_USD(currency)
        commission=commission * get_USD(currency)
        value=value * get_USD(currency) 
                   
    commission = max(commission, comCash)
    pl = purepl - commission       

    if abs(closeqty) > abs(openqty):
        openAt=closeAt
    
    side='long'
    if qty < 0:
        side='short'
    
    return (openAt, qty, commission, value, pl, mult, side)

def calc_closeVWAP(closeVWAP, closedqty, addVWAP, addqty):
    newVWAP = (abs(closeVWAP * closedqty) + abs(addVWAP*addqty)) / (abs(closedqty) + abs(addqty))
    newqty = abs(closedqty) + abs(addqty)
    return (newVWAP, newqty)
                
def calc_add_pos(openVWAP, openqty, addAt, addQty, comPct, comCash, currency, mult):
    newVWAP=(abs(openVWAP * openqty) + abs(addAt * addQty)) / (abs(openqty) + abs(addQty))
    newqty=openqty+addQty
    commission=abs(comPct * addAt * addQty * mult)               
    buy_power=abs(mult * addAt * addQty) * -1
    purepl=0
    if currency != 'USD':
            purepl=purepl * get_USD(currency)
            commission=commission * get_USD(currency)
            buy_power=buy_power * get_USD(currency)
    commission = max(commission, comCash)
    pl =  - commission      
    side='long'
    if newqty < 0:
        side='short'
    return (newVWAP, newqty, commission, buy_power, pl, mult, side)

def calc_pl(openAt, closeAt, qty, mult, side):
    
    if side == 'short':
        pl=(openAt - closeAt)*abs(qty)*abs(mult)
        value=closeAt*abs(qty)*abs(mult)
        return (pl, value)
        
    if side == 'long':
        pl=(closeAt - openAt)*abs(qty)*abs(mult)
        value=closeAt*abs(qty)*abs(mult)
        return (pl,value)
               
def get_c2_trades(systemname, date):
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    
    if os.path.isfile(filename):
        existData = pd.read_csv(filename, index_col='trade_id')
        return existData
    else:
        dataSet=pd.DataFrame({},columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
        'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
        'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
        'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        dataSet = dataSet.set_index('trade_id')
        dataSet.to_csv(filename)
        return dataSet
  
    return dataSet

def get_c2_portfolio(systemname, date):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    
    account=get_account_value(systemname, 'c2', date)
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        return (account, dataSet)
        
    else:
        dataSet=pd.DataFrame({},columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
        'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
        'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
        'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        dataSet = dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        return (account, dataSet)

def update_c2_portfolio(systemname, pos, date):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    (account, dataSet)=get_c2_portfolio(systemname, date)

    pos=pos.copy()
    tradeid=int(pos['trade_id'])
    if debug:
        print "Update Portfolio: " + str(tradeid)
    dataSet=dataSet.reset_index()
    pos['trade_id'] = pos['trade_id'].astype('int')
    symbol=pos['symbol']
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    
    if pos['quant_opened'] != pos['quant_closed']:
        if symbol in dataSet['symbol'].values:
        #if tradeid in dataSet['trade_id'].values:
            dataSet = dataSet[dataSet['trade_id'] != tradeid]
            dataSet = dataSet[dataSet['symbol'] != symbol]
            dataSet=dataSet.append(pos)
        else:
            dataSet=dataSet.append(pos)
    else:
        dataSet = dataSet[dataSet['trade_id'] != tradeid]
        dataSet = dataSet[dataSet['symbol'] != symbol]
    if debug:
        print "Update Portfolio " + systemname + " " + pos['long_or_short'] + \
                        ' symbol: ' + pos['symbol'] + ' open_or_closed ' + pos['open_or_closed'] + \
                        ' opened: ' + str(pos['quant_opened']) + ' closed: ' + str(pos['quant_closed'])
    dataSet=dataSet.set_index('symbol')
    dataSet.to_csv(filename)
    account=get_account_value(systemname, 'c2', date)
    return (account, dataSet)
    
def update_c2_trades(systemname, pos, tradepl, pl, buypower, date):
    
    account=get_account_value(systemname, 'c2', date)
    account['balance']=account['balance']+tradepl
    account['buy_power']=account['buy_power']+buypower
    account['real_pnl']=account['real_pnl'] + tradepl
    account['Date']=date
    account=update_account_value(systemname, 'c2', account)
    
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    dataSet = get_c2_trades(systemname, date)
    #pos=pos.iloc[-1]
    pos=pos.copy()
    tradeid=int(pos['trade_id'])
    dataSet=dataSet.reset_index()
    
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    pos['trade_id'] = pos['trade_id'].astype('int')
    
    pos['balance']=account['balance'] 
    pos['margin_available']=account['buy_power']
    
    if tradeid in dataSet['trade_id'].values:
        dataSet = dataSet[dataSet['trade_id'] != tradeid]
        dataSet=dataSet.append(pos)
         
    else:
        dataSet=dataSet.append(pos)
        
    if debug:
        print "Update Trade " + systemname + " " + pos['long_or_short'] + \
                    ' symbol: ' + pos['symbol'] + ' open_or_closed ' + pos['open_or_closed'] + \
                    ' opened: ' + str(pos['quant_opened']) + ' closed: ' + str(pos['quant_closed'])
    #print filename
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    dataSet=dataSet.set_index('trade_id')   
    dataSet.to_csv(filename)
    
    return (account, dataSet)
    
def get_c2_pos(systemname, c2sym, date):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    
    (account, dataSet)=get_c2_portfolio(systemname, date)

    if c2sym not in dataSet.index.values:
        dataSet= pd.DataFrame([[c2sym,0,0,'none']], \
                              columns=['symbol','quant_opened','quant_closed','long_or_short'])
        dataSet=dataSet.set_index('symbol')
        
    return dataSet    



def get_account_value(systemname, broker, date):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['Date'])
        
    else:
        dataSet=pd.DataFrame([[date, 'paperUSD',10000,30000,0,0,'USD']], columns=['Date','accountid','balance','buy_power','unr_pnl','real_pnl','currency'])
        dataSet=dataSet.set_index('Date')
        dataSet.to_csv(filename)
        
    account=pd.DataFrame([[date, dataSet.iloc[-1]['accountid'],
                            dataSet.iloc[-1]['balance'],dataSet.iloc[-1]['buy_power'],
                            dataSet.iloc[-1]['unr_pnl'], dataSet.iloc[-1]['real_pnl'], 
                            dataSet.iloc[-1]['currency']]], 
        columns=['Date','accountid','balance','buy_power','unr_pnl','real_pnl','currency']).iloc[-1]
        
    return account

def update_account_value(systemname, broker, account):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    old_account=get_account_value(systemname, broker, account['Date'])
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['Date'])
        dataSet=dataSet.reset_index()
        dataSet=dataSet.append(account)
        dataSet=dataSet.set_index('Date')
        dataSet.to_csv(filename)
    else:
        dataSet=get_account_value(systemname, broker, account['Date'])
        
    account=get_account_value(systemname, broker, account['Date'])
    return account
    

def get_ib_trades(systemname, date):
    filename='./data/paper/ib_' + systemname + '_trades.csv'
    
    if os.path.isfile(filename):
        existData = pd.read_csv(filename, index_col='permid')
        return existData
    else:
        dataSet=pd.DataFrame({}, columns=['permid','account','clientid','commission','commission_currency',\
                            'exchange','execid','expiry','level_0','orderid','price','qty','openqty', \
                            'realized_PnL','side',\
                            'symbol','symbol_currency','times','yield_redemption_date'])
        dataSet=dataSet.set_index('permid')
        dataSet.to_csv(filename)
        return dataSet
        
        
def get_ib_portfolio(systemname, date):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
    
    account=get_account_value(systemname, 'ib', date)
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        dataSet=dataSet.reset_index()
        dataSet['symbol']=dataSet['sym'] + dataSet['currency'] 
        dataSet=dataSet.set_index('symbol')
        return (account, dataSet)
    else:

        dataSet=pd.DataFrame({}, columns=['sym','exp','qty','openqty','price','openprice','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
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
       dataSet=pd.DataFrame([[sym_cur,symbol,0,0,currency]], \
                              columns=['symbol','sym','qty','openqty','currency'])
    
       dataSet=dataSet.set_index('symbol')
       return dataSet
    return portfolio_data
    
def update_ib_portfolio(systemname, pos, date):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
   
    (account, dataSet)=get_ib_portfolio(systemname, date)
    dataSet=dataSet.reset_index()
    
    pos=pos.copy()
    symbol = pos['sym'] + pos['currency']
    pos['qty']=pos['openqty']
    pos['price']=pos['openprice']
    pos['openprice']=pos['openprice']
    pos['symbol']=symbol
    dataSet['symbol']=dataSet['sym'] + dataSet['currency']
    if debug:
        print "Update Portfolio: " + str(symbol)

    if float(pos['qty']) != 0:
        
        if symbol in dataSet.loc[dataSet['symbol'] == symbol].values:
            dataSet = dataSet[dataSet['symbol'] != symbol]
            dataSet=dataSet.append(pos)
        else:
            dataSet=dataSet.append(pos)
            
    else:
        dataSet = dataSet[dataSet['symbol'] != symbol]
        
    if debug:
        print "Update Portfolio " + systemname + " Qty: " + str(pos['qty']) + \
                        ' symbol: ' + symbol
                        
    dataSet=dataSet.set_index('symbol')
    
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'ib', date)
    return (account, dataSet)
    
def update_ib_trades(systemname, pos, pl, buypower, ibexch, date):
    filename='./data/paper/ib_' + systemname + '_trades.csv'
   
    dataSet = get_ib_trades(systemname, date)
    #pos=pos.iloc[-1]
    pos=pos.copy()
    trade_id=0
    dataSet=dataSet.reset_index()
    if len(dataSet['permid'].values) > 0:
        trade_id=int(max(dataSet['permid'].values))
    trade_id=int(trade_id) + 1
    side='BOT'
    if pos['qty'] < 0:
        side='SLD'
    if debug:
        print "Trade ID:" + str(trade_id)
    
    pos=pd.DataFrame([[trade_id, 'Paper', 'Paper', pos['avg_cost'], 'USD', \
                               ibexch, trade_id, '','',1,pos['price'],abs(pos['qty']),pos['openqty'],pos['openprice'], \
                               pos['real_pnl'],side, \
                               pos['sym'], pos['currency'], date, '' \
                            ]], columns=['permid','account','clientid','commission','commission_currency',\
                            'exchange','execid','expiry','level_0','orderid','price','qty','openqty','openprice', \
                            'realized_PnL','side',\
                            'symbol','symbol_currency','times','yield_redemption_date']).iloc[-1]
                            
    tradeid=int(pos['permid'])
    
    dataSet['permid'] = dataSet['permid'].astype('int')
    pos['permid'] = pos['permid'].astype('int')
    
    account=get_account_value(systemname, 'ib', date)
    account['balance']=account['balance']+pl
    account['buy_power']=account['buy_power']+buypower
    account['real_pnl']=account['real_pnl'] + pl
    account['Date']=date
    account=update_account_value(systemname, 'ib',account)
    
    pos['balance']=account['balance'] 
    pos['margin_available']=account['buy_power']
    
    if tradeid in dataSet['permid'].values:
        dataSet = dataSet[dataSet['permid'] != tradeid]
        dataSet=dataSet.append(pos)
         
    else:
        dataSet=dataSet.append(pos)
        
    if debug:
        print "Update Trade " + systemname + " " + pos['side'] + \
                    ' symbol: ' + pos['symbol'] + ' qty: ' + str(pos['qty']) + \
                    ' openqty: ' + str(pos['openqty'])
    #print filename
    dataSet['permid'] = dataSet['permid'].astype('int')
    dataSet=dataSet.set_index('permid')   
    dataSet.to_csv(filename)
    

    return (account, dataSet)

def get_USD(currency):
    data=pd.read_csv('./data/systems/currency.csv')
    conversion=float(data.loc[data['Symbol']==currency].iloc[-1]['Ask'])
    return float(conversion)