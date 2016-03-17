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

def adj_size(model_pos, system, system_name, pricefeed, c2systemid, c2apikey, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit, date):
    system_pos=model_pos.loc[system]
   
    c2submit=True
    ibsubmit=True
    if c2submit:
        c2_pos=get_c2_pos(system_name, c2sym)
        c2_pos=c2_pos.loc[c2_pos['symbol']==c2sym].iloc[-1]
        c2_pos_qty=int(c2_pos['quant_opened']) - int(c2_pos['quant_closed'])
        c2_pos_side=str(c2_pos['long_or_short'])
        if c2_pos_side == 'short':
            c2_pos_qty=-c2_pos_qty
            
        system_c2pos_qty=round(system_pos['action']) * c2quant
        
        if system_c2pos_qty != c2_pos_qty:
            print "========"
            print "system: " + system_name + " symbol: " + c2_pos['symbol']
            print "system_c2_pos: " + str(system_c2pos_qty)
            print "c2_pos: " + str(c2_pos_qty)
            print "========"
        
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'BTC: ' + str(qty)
                place_order(system_name, 'BTC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                                
                c2quant = c2quant - qty
            
            if c2quant > 0:
                print 'BTO: ' + str(c2quant)
                place_order(system_name, 'BTO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'STC: ' + str(qty)
                place_order(system_name, 'STC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
                c2quant = c2quant - qty

            if c2quant > 0:
                print 'STO: ' + str(c2quant)
                place_order(system_name, 'STO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed,date)
                
    if ibsubmit:
        paper_pos=get_ib_pos(system_name, ibsym,ibcurrency)
        symbol=ibsym+ibcurrency
        ib_pos=paper_pos.loc[paper_pos['symbol']==symbol].iloc[-1]
        ib_pos_qty=ib_pos['qty']
        system_ibpos_qty=round(system_pos['action']) * ibquant
        
        if system_ibpos_qty != ib_pos_qty:
            print "========"
            print "system: " + system_name + " symbol: " + ib_pos['symbol']
            print "system_ib_pos: " + str(system_ibpos_qty)
            print "ib_pos: " + str(ib_pos_qty)
            print "========"
        
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            print 'BUY: ' + str(ibquant)
            place_order(system_name, 'BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed,date);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            print 'SELL: ' + str(ibquant)
            place_order(system_name, 'SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed,date)

def test(sym, systemname):
    (account, portfolio)=get_c2_portfolio(systemname)
    if sym in portfolio['symbol'].values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[portfolio['symbol']==sym].reset_index().iloc[-1]
            print pos['trade_id']

def place_order(systemname, action, quant, sym, type, currency, exch, broker, pricefeed, date):
    print "Place Order " + action + " " + str(quant) + " " + sym + " " + currency + " " + broker 
    pricefeed=pricefeed.iloc[-1]
    
    if broker == 'c2':
        (account, portfolio)=get_c2_portfolio(systemname)
    
            
        if sym in portfolio['symbol'].values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[portfolio['symbol']==sym].reset_index().iloc[-1]
            
            side=str(pos['long_or_short'])
            openVWAP=float(pos['opening_price_VWAP'])
            closeVWAP=float(pos['closing_price_VWAP'])
            
            openqty=int(pos['quant_opened'])
            closedqty=int(pos['quant_closed'])
            openVWAP_timestamp=str(pos['openVWAP_timestamp'])
            closeVWAP_timestamp=str(pos['closeVWAP_timestamp'])
            pl=float(pos['PL'])
            ptValue=float(pos['ptValue'])
            
            print 'Symbol: ' + pos['symbol'] + ' Action: ' + action +' Side: ' + side
            if action == 'STO' and side=='short':
                bid=pricefeed['Bid']
                
                print "C2 STO " + str(sym) + "@" + str(bid) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                        
                openVWAP=(openVWAP * openqty + bid * quant) / (openqty + quant)
                openqty=openqty + quant
                pos['openVWAP_timestamp']=date
                pos['opening_price_VWAP']=openVWAP
                pos['quant_opened']=openqty
                
                commission=abs(pricefeed['Commission_Pct'] * bid * quant * ptValue)
                purepl=0
                
                buy_power=pos['ptValue'] * openVWAP * quant * -1
                
                print "Original PL: " + str(pos['PL']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"
                
                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                    
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl + purepl - commission 
                
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission 
                
                update_c2_trades(systemname, pos, pl, buy_power)
                update_c2_portfolio(systemname, pos)
                
            if action == 'BTO' and side=='long':
                ask=pricefeed['Ask']
                
                print "C2 BTO " + str(sym) + "@" + str(ask) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                        
                openVWAP=(openVWAP * openqty + ask * quant) / (openqty + quant)
                openqty=openqty + quant
                pos['openVWAP_timestamp']=date
                pos['opening_price_VWAP']=openVWAP
                pos['quant_opened']=openqty
                commission=(abs(pricefeed['Commission_Pct'] * ask * quant * ptValue)) 
                purepl=0
                
                buy_power=pos['ptValue'] * openVWAP * quant * -1
                
                print "Original PL: " + str(pos['PL']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"
                
                if currency != 'USD':
                    purepl = purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                    
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl + purepl - commission                     
                    
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission
                
                update_c2_trades(systemname, pos, pl, buy_power)
                update_c2_portfolio(systemname, pos)

            if action == 'STC' and side=='long':
                bid=pricefeed['Bid']
                
                closeVWAP = (closeVWAP * closedqty + bid * quant) / (closedqty + quant)
                closedqty = closedqty + quant
                closeVWAP_timestamp=date
                commission=(abs(pricefeed['Commission_Pct'] * bid * quant * ptValue)) 
                
                purepl=(openVWAP - closeVWAP) * quant * pricefeed['C2Mult']
                
                pos['closeVWAP_timestamp']=date
                pos['closedWhen']=date
                pos['closedWhenUnixTimeStamp']=date
                pos['closing_price_VWAP']=closeVWAP
                pos['quant_closed']=closedqty

                if closedqty == openqty:
                    pos['open_or_closed']='closed'
                    
                buy_power=pos['ptValue'] * closeVWAP * quant
               
                print "C2 STC " + str(sym) + "@" + str(bid) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                print "Original PL: " + str(pos['PL']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "Trade PL: " + str(purepl)
                print "Close VWAP: " + str(closeVWAP) + " [" + str(closedqty) + "]"

                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl + purepl - commission                  
                    
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission 
                
                update_c2_trades(systemname, pos, pl, buy_power)
                update_c2_portfolio(systemname, pos)
                    
                
            if action == 'BTC' and side=='short':
                ask=pricefeed['Ask']
                closeVWAP = (closeVWAP * closedqty + ask * quant) / (closedqty + quant)
                closedqty = closedqty + quant
                closeVWAP_timestamp=date
                commission=(abs(pricefeed['Commission_Pct'] * ask * quant * ptValue)) 
                
                purepl = (closeVWAP - openVWAP) * quant * pricefeed['C2Mult']
                
                pos['closeVWAP_timestamp']=date
                pos['closedWhen']=date
                pos['closedWhenUnixTimeStamp']=date
                pos['closing_price_VWAP']=closeVWAP
                pos['quant_closed']=closedqty
                    
                if closedqty == openqty:
                    pos['open_or_closed']='closed'
                    
                buy_power=pos['ptValue'] * closeVWAP * quant
                print "C2 BTC " + str(sym) + "@" + str(ask) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                print "Original PL: " + str(pos['PL']) + " New PL: " + str(pl) + " (Commission: " + str(commission) + ")"
                print "Trade PL: " + str(purepl)
                print "Close VWAP: " + str(closeVWAP) + " [" + str(closedqty) + "]"

                if currency != 'USD':
                    purepl = purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                    
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl + purepl - commission          
                    
                pos['PL']=pl
                pos['commission']=pos['commission'] + commission
                update_c2_trades(systemname, pos, pl, buy_power)
                update_c2_portfolio(systemname, pos)
                
                #data=pd.DataFrame([[trade_id, 0, '','', \
                #                '',0,'',sym,side, \
                #                 date,date,'open',date,openVWAP, \
                #                10000,'',0,openqty,'',sym,systemname]] \
                #,columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
                #'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
                #'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
                #'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        elif action == 'STO' or action == 'BTO':
            side='long'
            openVWAP=-1
            if action == 'STO':
                side='short'
                openVWAP=pricefeed['Bid']
                
            if action == 'BTO':
                side='long'
                openVWAP=pricefeed['Ask']
                
            trade_id=0
            trades=get_c2_trades(systemname)
            if len(trades.index.values) > 0:
                trade_id=int(max(trades.index.values))
            trade_id=int(trade_id) + 1
            print "Trade ID:" + str(trade_id)
            
            openqty=quant
            ptValue=pricefeed['C2Mult']
            commission=(abs(pricefeed['Commission_Pct'] * openVWAP * openqty * ptValue) ) 
            pl=0
            buy_power=openVWAP * openqty * ptValue * -1

            if currency != 'USD':
                    pl=pl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
            
            commission = max(commission,pricefeed['Commission_Cash'])
            pl = pl - commission                    
       
            
            pos=pd.DataFrame([[trade_id, pl, '','', \
                                '',0,'',sym,side, \
                                date,date,'open',date,openVWAP, \
                                ptValue,'',0,openqty,'',sym,systemname, \
                                commission]] \
            ,columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
            'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
            'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
            'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description', \
            'commission']).iloc[-1]
            update_c2_trades(systemname, pos, pl, buy_power)
            update_c2_portfolio(systemname, pos)
            
    if broker == 'ib':
        (account, portfolio)=get_ib_portfolio(systemname)
        
        symbol=sym+currency
        portfolio=portfolio.reset_index()
        if symbol in portfolio.loc[portfolio['symbol']==symbol].values:
            #portfolio=portfolio.reset_index().set_index('symbol')
            pos=portfolio.loc[portfolio['symbol']==symbol].iloc[-1]
            
            openVWAP=float(pos['price'])
            openqty=int(pos['openqty'])
            side='long'
            if openqty < 0:
                side='short'
                
            pl=0
            ptValue=float(pricefeed['IBMult'])
            
            print 'Symbol: ' + pos['sym'] + pos['currency'] + ' Action: ' + action +' Side: ' + side
            if action == 'SELL' and side=='short':
                bid=pricefeed['Bid']
                quant=-quant
                
                print "IB STO " + str(symbol) + "@" + str(bid) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                        
                openVWAP=(openVWAP * abs(openqty) + bid * abs(quant)) / (abs(openqty) + abs(quant))
                
                openqty=openqty + quant
                buy_power=pricefeed['IBMult'] * openVWAP * abs(quant) * -1
                commission=(abs(pricefeed['Commission_Pct'] * bid * abs(quant) * ptValue)) 
                purepl = 0
                
                #pos['times']=date
                value=pos['value']
                pos['price']=openVWAP
                pos['qty']=quant
                pos['value']=abs(pricefeed['IBMult'] * openVWAP * openqty)
                
                print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "Original Value: " + str(value) + " New Value: " + str(pos['value'])     
                print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"

                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                    
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl +purepl - commission             
                
                pos['avg_cost']=commission 
                pos['real_pnl']=pl
                pos['unr_pnl']=0
                pos['openqty']=openqty
                
                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos)
                
            if action == 'BUY' and side == 'long':
                ask=pricefeed['Ask']
                
                print "IB BTO " + str(symbol) + "@" + str(ask) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                        
                openVWAP=(openVWAP * abs(openqty) + ask * abs(quant)) / (abs(openqty) + abs(quant))
                openqty=openqty + quant
                
                buy_power=pricefeed['IBMult'] * openVWAP * abs(quant) * -1
                commission=(abs(pricefeed['Commission_Pct'] * ask * abs(quant) * ptValue)) 
                purepl=0
                
                #pos['times']=date
                value=pos['value']
                pos['price']=openVWAP
                pos['qty']=quant
                pos['value']=abs(pricefeed['IBMult'] * openVWAP * openqty)
                
                print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "Original Value: " + str(value) + " New Value: " + str(pos['value'])     
                print "New VWAP: " + str(openVWAP) + " [" + str(openqty) + "]"

                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl +purepl - commission         
                
                pos['avg_cost']=commission
                pos['real_pnl']=pl
                pos['unr_pnl']=0
                pos['openqty']=openqty
                
                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos)


            if action == 'SELL' and side=='long':
                bid=pricefeed['Bid']
                quant=-quant
                
                newVWAP = (openVWAP * abs(openqty) + bid * abs(quant)) / (abs(openqty) + abs(quant))
                newqty = openqty + quant
                newVWAP_timestamp=date
                commission=(abs(pricefeed['Commission_Pct'] * bid * abs(quant) * ptValue)) 
                buy_power=pricefeed['IBMult'] * newVWAP * quant
                
                print "NQ: " + str(newqty) + " Q: " + str(quant)
                
                purepl=(openVWAP - newVWAP) * abs(quant) * pricefeed['IBMult']
                
                #pos['times']=newVWAP_timestamp
                value=pos['value']
                pos['price']=newVWAP
                pos['qty']=quant
                pos['value']=abs(pricefeed['IBMult'] * newVWAP * newqty)
                
                print "IB SELL " + str(symbol) + "@" + str(bid) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "Trade PL: " + str(purepl)
                print "Original Value: " + str(value) + " New Value: " + str(pos['value'])            
                print "New VWAP: " + str(newVWAP) + " [" + str(newqty) + "]"

                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl +purepl - commission         
                
                pos['real_pnl']=pl
                pos['avg_cost']=commission 
                pos['unr_pnl']=0
                pos['openqty']=newqty
                
                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos)
                    
                
            if action == 'BUY' and side=='short':
                ask=pricefeed['Ask']
                newVWAP = (openVWAP * abs(openqty) + ask * abs(quant)) / (abs(openqty) + abs(quant))
                newqty = openqty + quant
                
                newVWAP_timestamp=date
                commission=(abs(pricefeed['Commission_Pct'] * ask * abs(quant) * ptValue)) 
                buy_power=pricefeed['IBMult'] * newVWAP * quant
                
                purepl=(newVWAP - openVWAP) * abs(quant) * pricefeed['IBMult']

                value=pos['value']
                pos['price']=newVWAP
                pos['qty']=quant
                pos['value']=abs(pricefeed['IBMult'] * newVWAP * newqty)
                
                print "IB BUY " + str(symbol) + "@" + str(ask) + "[" + str(quant) + "] Opened @ " + \
                        str(openVWAP) + "[" + str(openqty) + "]" 
                print "Original PL: " + str(pos['real_pnl']) + " New PL: " + str(purepl) + " (Commission: " + str(commission) + ")"
                print "Trade PL: " + str(purepl)
                print "Original Value: " + str(value) + " New Value: " + str(pos['value'])     
                print "New VWAP: " + str(newVWAP) + " [" + str(newqty) + "]"

                if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency) 
                commission = max(commission,pricefeed['Commission_Cash'])
                pl = pl +purepl - commission         
                
                pos['real_pnl']=pl
                pos['avg_cost']=commission
                pos['unr_pnl']=0
                pos['openqty']=newqty
               
                update_ib_trades(systemname, pos, pl, buy_power, exch, date)
                update_ib_portfolio(systemname, pos)
                    
            
        else:
            side='long'
            openVWAP=-1
            if action == 'SELL':
                side='short'
                openVWAP=pricefeed['Bid']
                quant=-quant
                
            if action == 'BUY':
                side='long'
                openVWAP=pricefeed['Ask']
                
            openqty=quant
            ptValue=pricefeed['IBMult']
            value=abs(openVWAP * openqty * ptValue)
            commission=(abs(pricefeed['Commission_Pct'] * openVWAP * abs(openqty) * ptValue))
            purepl=0
            buy_power=openVWAP * openqty * ptValue * -1
            
            if currency != 'USD':
                    purepl=purepl * get_USD(currency)
                    commission=commission * get_USD(currency)
                    buy_power=buy_power * get_USD(currency)
            commission = max(commission,pricefeed['Commission_Cash'])
            pl =  - commission                     
             
            pos=pd.DataFrame([[sym,'',openqty,openVWAP, value, \
                                    commission, 0, pl, 'Paper', \
                                    currency,openqty]], 
                                 columns=['sym','exp','qty','price','value', \
                                 'avg_cost','unr_pnl','real_pnl','accountid', \
                                 'currency', 'openqty']).iloc[-1]
            
            update_ib_trades(systemname, pos, pl, buy_power, exch, date)
            update_ib_portfolio(systemname, pos)   

def get_c2_trades(systemname):
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    
    datestr=strftime("%Y%m%d", localtime())
    
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

def get_c2_portfolio(systemname):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
    account=get_account_value(systemname, 'c2')
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['trade_id'])
        return (account, dataSet)
        
    else:
        dataSet=pd.DataFrame({},columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
        'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
        'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
        'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        dataSet = dataSet.reset_index().set_index(['trade_id'])
        dataSet.to_csv(filename)
        return (account, dataSet)

def update_c2_portfolio(systemname, pos):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    (account, dataSet)=get_c2_portfolio(systemname)

    pos=pos.copy()
    tradeid=int(pos['trade_id'])
    print "Update Portfolio: " + str(tradeid)
    dataSet=dataSet.reset_index()
    pos['trade_id'] = pos['trade_id'].astype('int')
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    if str(pos['open_or_closed']) == 'open':
        
        if tradeid in dataSet['trade_id'].values:
            dataSet = dataSet[dataSet['trade_id'] != tradeid]
            dataSet=dataSet.append(pos)
        else:
            dataSet=dataSet.append(pos)
            
    else:
        dataSet = dataSet[dataSet['trade_id'] != tradeid]
    print "Update Portfolio " + systemname + " " + pos['long_or_short'] + \
                        ' symbol: ' + pos['symbol'] + ' open_or_closed ' + pos['open_or_closed'] + \
                        ' opened: ' + str(pos['quant_opened']) + ' closed: ' + str(pos['quant_closed'])
    dataSet=dataSet.set_index('trade_id')
    
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'c2')
    return (account, dataSet)
    
def update_c2_trades(systemname, pos, pl, buypower):
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    dataSet = get_c2_trades(systemname)
    #pos=pos.iloc[-1]
    pos=pos.copy()
    tradeid=int(pos['trade_id'])
    dataSet=dataSet.reset_index()
    
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    pos['trade_id'] = pos['trade_id'].astype('int')
    
    if tradeid in dataSet['trade_id'].values:
        dataSet = dataSet[dataSet['trade_id'] != tradeid]
        dataSet=dataSet.append(pos)
         
    else:
        dataSet=dataSet.append(pos)
        
    print "Update Trade " + systemname + " " + pos['long_or_short'] + \
                    ' symbol: ' + pos['symbol'] + ' open_or_closed ' + pos['open_or_closed'] + \
                    ' opened: ' + str(pos['quant_opened']) + ' closed: ' + str(pos['quant_closed'])
    #print filename
    dataSet['trade_id'] = dataSet['trade_id'].astype('int')
    dataSet=dataSet.set_index('trade_id')   
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'c2')
    account['balance']=account['balance']+pl
    account['buy_power']=account['buy_power']+buypower
    account['real_pnl']=account['real_pnl'] + pl
    account=update_account_value(systemname, 'c2',account)
    return (account, dataSet)
    
def get_c2_pos(systemname, c2sym):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    (account, dataSet)=get_c2_portfolio(systemname)

    if c2sym not in dataSet['symbol'].values:
        return pd.DataFrame([[c2sym,0,0,'none']], \
                              columns=['symbol','quant_opened','quant_closed','long_or_short'])
    return dataSet    



def get_account_value(systemname, broker):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['accountid'])
        
    else:
        dataSet=pd.DataFrame([['paperUSD',10000,30000,0,0,'USD']], columns=['accountid','balance','buy_power','unr_pnl','real_pnl','currency'])
        dataSet=dataSet.set_index('accountid')
        dataSet.to_csv(filename)
    return dataSet

def update_account_value(systemname, broker, account):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    account.to_csv(filename)
    return account
    

def get_ib_trades(systemname):
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
        
        
def get_ib_portfolio(systemname):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
    account=get_account_value(systemname, 'ib')
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['sym','currency'])
        dataSet=dataSet.reset_index()
        dataSet['symbol']=dataSet['sym'] + dataSet['currency'] 
        dataSet=dataSet.set_index(['sym','currency'])
        return (account, dataSet)
    else:

        dataSet=pd.DataFrame({}, columns=['sym','exp','qty','openqty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
        dataSet['symbol']=dataSet['sym'] + dataSet['currency']        
        dataSet=dataSet.set_index(['sym','currency'])
        dataSet.to_csv(filename)
        return (account, dataSet)
   
def get_ib_pos(systemname, symbol, currency):
    datestr=strftime("%Y%m%d", localtime())
    (account_data, portfolio_data)=get_ib_portfolio(systemname)
    portfolio_data=portfolio_data.reset_index()
    portfolio_data['symbol']=portfolio_data['sym'] + portfolio_data['currency']
    sym_cur=symbol + currency
    if sym_cur not in portfolio_data['symbol'].values:
       return pd.DataFrame([[sym_cur,symbol,0,0,currency]], \
                              columns=['symbol','sym','qty','openqty','currency'])
    
    
    return portfolio_data
    
def update_ib_portfolio(systemname, pos):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    (account, dataSet)=get_ib_portfolio(systemname)
    dataSet=dataSet.reset_index()
    
    pos=pos.copy()
    symbol = pos['sym'] + pos['currency']
    pos['qty']=pos['openqty']
    dataSet['symbol']=dataSet['sym'] + dataSet['currency']
    print "Update Portfolio: " + str(symbol)

    if int(pos['qty']) != 0:
        
        if symbol in dataSet.loc[dataSet['symbol'] == symbol].values:
            dataSet = dataSet[dataSet['symbol'] != symbol]
            dataSet=dataSet.append(pos)
        else:
            dataSet=dataSet.append(pos)
            
    else:
        dataSet = dataSet[dataSet['symbol'] != symbol]
        
    print "Update Portfolio " + systemname + " Qty: " + str(pos['qty']) + \
                        ' symbol: ' + symbol
                        
    dataSet=dataSet.set_index(['sym','currency'])
    
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'ib')
    return (account, dataSet)
    
def update_ib_trades(systemname, pos, pl, buypower, ibexch, date):
    filename='./data/paper/ib_' + systemname + '_trades.csv'
   
    dataSet = get_ib_trades(systemname)
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
    print "Trade ID:" + str(trade_id)
    
    pos=pd.DataFrame([[trade_id, 'Paper', 'Paper', pos['avg_cost'], 'USD', \
                               ibexch, trade_id, '','',1,pos['price'],abs(pos['qty']),pos['openqty'], \
                               pos['real_pnl'],side, \
                               pos['sym'], pos['currency'], date, '' \
                            ]], columns=['permid','account','clientid','commission','commission_currency',\
                            'exchange','execid','expiry','level_0','orderid','price','qty','openqty', \
                            'realized_PnL','side',\
                            'symbol','symbol_currency','times','yield_redemption_date']).iloc[-1]
                            
    tradeid=int(pos['permid'])
    
    dataSet['permid'] = dataSet['permid'].astype('int')
    pos['permid'] = pos['permid'].astype('int')
    
    if tradeid in dataSet['permid'].values:
        dataSet = dataSet[dataSet['permid'] != tradeid]
        dataSet=dataSet.append(pos)
         
    else:
        dataSet=dataSet.append(pos)
        
    print "Update Trade " + systemname + " " + pos['side'] + \
                    ' symbol: ' + pos['symbol'] + ' qty: ' + str(pos['qty']) + \
                    ' openqty: ' + str(pos['openqty'])
    #print filename
    dataSet['permid'] = dataSet['permid'].astype('int')
    dataSet=dataSet.set_index('permid')   
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'ib')
    account['balance']=account['balance']+pl
    account['buy_power']=account['buy_power']+buypower
    account['real_pnl']=account['real_pnl'] + pl
    
    account=update_account_value(systemname, 'ib',account)
    return (account, dataSet)

def get_USD(currency):
    data=pd.read_csv('./data/systems/currency.csv')
    conversion=float(data.loc[data['Symbol']==currency].iloc[-1]['Ask'])
    return float(conversion)