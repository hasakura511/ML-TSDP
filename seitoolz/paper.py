import numpy as np
import pandas as pd
import time
import os.path

import json
from pandas.io.json import json_normalize
from seitoolz.signal import get_model_pos
from time import gmtime, strftime, time, localtime, sleep
    
def adj_size(model_pos, system, system_name, pricefeed, c2systemid, c2apikey, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    system_pos=model_pos.loc[system]
   
    print "system: " + system
    #print str(system_pos['action'])
    #print "c2: " 
    #print c2_pos
    c2submit=True
    ibsubmit=True
    if c2submit:
        c2_pos=get_c2_pos(system_name, c2sym).loc[c2sym]
        c2_pos_qty=int(c2_pos['quant_opened']) - int(c2_pos['quant_closed'])
        c2_pos_side=c2_pos['long_or_short']
        if c2_pos_side == 'short':
            c2_pos_qty=-c2_pos_qty
            
        system_c2pos_qty=round(system_pos['action']) * c2quant
        print "system_c2_pos: " + str(system_c2pos_qty)
        print "c2_pos: " + str(c2_pos_qty)
        
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'BTC: ' + str(qty)
                place_order('BTC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed)
                                
                c2quant = c2quant - qty
                sleep(2)
            
            if c2quant > 0:
                print 'BTO: ' + str(c2quant)
                place_order('BTO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed)
                
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'STC: ' + str(qty)
                place_order('STC', qty, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed)
                
                c2quant = c2quant - qty
                sleep(2)
            if c2quant > 0:
                print 'STO: ' + str(c2quant)
                place_order('STO', c2quant, c2sym, c2type, ibcurrency, ibexch, 'c2', pricefeed)
                
    if ibsubmit:
        paper_pos=get_ib_pos(system_name, ibsym,ibcurrency)
        ib_pos=paper_pos.loc[ibsym,ibcurrency]
        ib_pos_qty=ib_pos['qty']
        system_ibpos_qty=round(system_pos['action']) * ibquant
        print "system_ib_pos: " + str(system_ibpos_qty)
        print "ib_pos: " + str(ib_pos_qty)
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            print 'BUY: ' + str(ibquant)
            place_order('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            print 'SELL: ' + str(ibquant)
            place_order('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, 'ib', pricefeed);

def place_order(systemname, action, quant, sym, type, currency, exch, broker, pricefeed):
    if broker == 'c2':
        portfolio=get_c2_portfolio(systemname)
        
        eastern=timezone('US/Eastern')
        endDateTime=dt.now(get_localzone())
        date=endDateTime.astimezone(eastern)
        date=date.strftime("%Y%m%d %H:%M:%S EST")
            
        trade_id=int(portfolio['trade_id'].values.max())
        trade_id=trade_id + 1
        if sym in portfolio['symbol'].values:
            pos=portfolio.loc[sym]
            
            side='long'
            openVWAP=-1
            openqty=quant
            
            if action == 'STO':
                side='short'
                openVWAP=pricefeed['Bid'][0]
                
            if action == 'BTO':
                side='long'
                openVWAP=pricefeed['Ask'][0]
            
            if action == 'STC':
            if action == 'BTC':
            data=pd.DataFrame([[trade_id, 0, '','', \
                                '','','',sym,side, \
                                date,date,'open',date,openVWAP, \
                                10000,'',0,openqty,'',sym,systemname]] \
            ,columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
            'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
            'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
            'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        else:
            side='long'
            openVWAP=-1
            openqty=quant
            
            if action == 'STO':
                side='short'
                openVWAP=pricefeed['Bid'][0]
                
            if action == 'BTO':
                side='long'
                openVWAP=pricefeed['Ask'][0]
            
            data=pd.DataFrame([[trade_id, 0, '','', \
                                '','','',sym,side, \
                                date,date,'open',date,openVWAP, \
                                10000,'',0,openqty,'',sym,systemname]] \
            ,columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
            'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
            'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
            'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
            portfolio.append(data)
            
    
    if broker == 'ib':
        data=pd.DataFrame({}, columns=['permid','account','clientid','commission','commission_currency',\
                        'exchange','execid','expiry','level_0','orderid','price','qty','realized_PnL','side',\
                        'symbol','symbol_currency','times','yield_redemption_date'])    
    #if action == 'BUY':
        #
    #if action == 'SELL':
        #
    #if action == 'BTO':
    #if action == 'BTC':
    #if action == 'STO':
    #if action == 'STC':
    print "Place Order"

def proc_exec(broker, sym):
    
    
def get_c2trades(systemid, systemname, c2api):
    filename='./data/paper/c2_' + systemname + '_trades.csv'
    
    datestr=strftime("%Y%m%d", localtime())
    #data=get_c2exec(systemid,c2api);
    #jsondata = json.loads(data)
    #dataSet=json_normalize(jsondata['response'])
    #dataSet=dataSet.set_index('trade_id')
     
    if os.path.isfile(filename):
        existData = pd.read_csv(filename, index_col='trade_id')
        existData = existData.reset_index()
        #dataSet   =   dataSet.reset_index()
        #dataSet=existData.append(dataSet)
        dataSet['trade_id'] = dataSet['trade_id'].astype('int')
        dataSet=dataSet.drop_duplicates(subset=['trade_id'],keep='last')
        dataSet=dataSet.set_index('trade_id') 
        
    dataSet=dataSet.sort_values(by='closedWhenUnixTimeStamp')
    
    #dataSet.to_csv(filename)
    return dataSet

def get_ibtrades(systemname):
    filename='./data/paper/ib_' + systemname + '_trades.csv'
    
    #datestr=strftime("%Y%m%d", localtime())
    #data=get_ibexec()
    #dataSet=pd.DataFrame(data)
    #dataSet=dataSet.set_index('permid')
    
    if os.path.isfile(filename):
        existData = pd.read_csv(filename, index_col='permid')
        existData =existData.reset_index()
        dataSet=dataSet.reset_index()
        dataSet=existData.append(dataSet)
        dataSet['permid'] = dataSet['permid'].astype('int')
        dataSet=dataSet.drop_duplicates(subset=['permid'],keep='last')
        dataSet=dataSet.set_index('permid')
    dataSet=dataSet.sort_values(by='times')
    #dataSet.to_csv(filename)
    return dataSet

def get_c2_portfolio(systemname):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['symbol'])
        
    else:
        dataSet=pd.DataFrame({},columns=['trade_id','PL','closeVWAP_timestamp','closedWhen',\
        'closedWhenUnixTimeStamp','closing_price_VWAP','expir','instrument','long_or_short',\
        'markToMarket_time','openVWAP_timestamp','open_or_closed','openedWhen','opening_price_VWAP',\
        'ptValue','putcall','quant_closed','quant_opened','strike','symbol','symbol_description'])
        dataSet = dataSet.reset_index().set_index(['symbol'])
        dataSet.to_csv(filename)
    account=get_account_value(systemname, 'c2')
    return (account, dataSet)

def update_c2_portfolio(systemname, portfolio):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    dataSet=portfolio
    dataSet = dataSet.reset_index().set_index(['symbol'])
    dataSet.to_csv(filename)
    
    account=get_account_value(systemname, 'c2')
    return (account, dataSet)
    
def get_c2_pos(systemname, c2sym):
    filename='./data/paper/c2_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    (account, dataSet)=get_c2_portfolio(systemname)
    dataSet=dataSet.reset_index()
    if c2sym not in dataSet['symbol'].values:
        dataSet=dataSet.append(pd.DataFrame([[c2sym,0,0,'none']], \
                              columns=['symbol','quant_opened','quant_closed','long_or_short']))
    dataSet=dataSet.set_index('symbol')
    return dataSet    

def get_ib_portfolio(systemname):
    filename='./data/paper/ib_' + systemname + '_portfolio.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['sym','currency'])
        
    else:
        dataSet=pd.DataFrame({}, columns=['sym','exp','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
        dataSet.to_csv(filename)

    account=get_account_value(systemname, 'ib')
    return (account, dataSet)

    
def get_ib_pos(systemname, symbol, currency):
    datestr=strftime("%Y%m%d", localtime())
    (account_data, portfolio_data)=get_ib_portfolio(systemname)
    portfolio_data=portfolio_data.reset_index()
    portfolio_data['symbol']=portfolio_data['sym'] + portfolio_data['currency']
    sym_cur=symbol + currency
    if sym_cur not in portfolio_data['symbol'].values:
        portfolio_data=portfolio_data.append(pd.DataFrame([[sym_cur,symbol,0,currency]], \
                              columns=['symbol','sym','qty','currency']))
    dataSet=portfolio_data
    #dataSet=dataSet.sort_values(by='times')
    dataSet=dataSet.set_index(['sym','currency'])
    #dataSet.to_csv('./data/portfolio/ib_portfolio.csv')
    #accountSet=pd.DataFrame(account_value)
    #accountSet.to_csv('./data/portfolio/ib_account_value.csv', index=False)
    #
    return dataSet

def get_account_value(systemname, broker):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    datestr=strftime("%Y%m%d", localtime())
   
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col=['accountid'])
        
    else:
        dataSet=pd.DataFrame([['paperUSD',10000,30000,0,0,'USD']], columns=['accountid','balance','buy_power','unr_pnl','real_pnl','currency'])
        
        dataSet.to_csv(filename)
    return dataSet