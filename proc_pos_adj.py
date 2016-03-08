import numpy as np
import pandas as pd
import time

import json
from pandas.io.json import json_normalize
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order
from ibapi.get_exec import get_exec_open as get_ibexec_open
from c2api.get_exec import get_exec_open as get_c2exec_open
from time import gmtime, strftime, time, localtime, sleep

def get_model_pos(systems):
    pos=pd.DataFrame({}, columns=['system','action','qty']);
    for system in systems:
        data = pd.read_csv('./data/signals/' + system + '.csv', index_col='Date')
        signals=data['Signal'];
        #safef=data['safef'];
       
        signal=signals[-1];
        qty=1
        pos=pos.append(pd.DataFrame([[system, signal, qty]], columns=['system','action','qty']))
        #pos=pos.append({'sym':system, 'action':signal, 'qty':qty}, ignore_index=True)
            
    pos=pos.set_index(['system'])
    
    pos.to_csv('./test.csv')
    return pos;
    
def adj_size(model_pos, ib_pos, system, systemid, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    system_pos=model_pos.loc[system]
   
    print "system: " + system
    #print str(system_pos['action'])
    #print "c2: " 
    #print c2_pos
    
    if c2submit:
        c2_pos=get_c2pos(systemid,system).loc[c2sym]
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
                place_c2order('BTC', qty, c2sym, c2type, systemid, c2submit)
                
                c2quant = c2quant - qty
                sleep(5)
            
            if c2quant > 0:
                print 'BTO: ' + str(c2quant)
                place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit)
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'STC: ' + str(qty)
                place_c2order('STC', qty, c2sym, c2type, systemid, c2submit)
                
                c2quant = c2quant - qty
                sleep(5)
            if c2quant > 0:
                print 'STO: ' + str(c2quant)
                place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit)
    if ibsubmit:
        ib_pos=ib_pos.loc[ibsym,ibcurrency]
        ib_pos_qty=ib_pos['qty']
        system_ibpos_qty=round(system_pos['action']) * ibquant
        print "system_ib_pos: " + str(system_ibpos_qty)
        print "ib_pos: " + str(ib_pos_qty)
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            print 'BUY: ' + str(ibquant)
            place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            print 'SELL: ' + str(ibquant)
            place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    #
    #place_iborder(ibaction, ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);

def get_c2pos(systemid, system):
    datestr=strftime("%Y%m%d", localtime())
    data=get_c2exec_open(systemid);
    
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        dataSet=dataSet.set_index(['symbol'])
        return dataSet
    else:
        dataSet=pd.DataFrame([[system,0,0,'none']],
                              columns=['symbol','quant_opened','quant_closed','long_or_short'])
        dataSet=dataSet.set_index(['symbol'])
        return dataSet
def get_ibpos():
    datestr=strftime("%Y%m%d", localtime())
    (account_value, portfolio_data)=get_ibexec_open()
    data=pd.DataFrame(portfolio_data,columns=['sym','exp','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
    dataSet=pd.DataFrame(data)
    #dataSet=dataSet.sort_values(by='times')
    dataSet=dataSet=dataSet.set_index(['sym','currency'])
    dataSet.to_csv('./data/ib_portfolio.csv')
    #
    return dataSet

model_pos=get_model_pos(['EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
ib_pos=get_ibpos()

adj_size(model_pos, ib_pos,'EURUSD','100961226',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100962402',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100962399',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100962390',2, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100962756',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100962769',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','100961267',1, 'EURUSD','forex',False, 25000,'EUR','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'GBPUSD','100961267',1, 'GBPUSD','forex',False, 25000,'GBP','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDCHF','100961267',1, 'USDCHF','forex',False, 25000,'USD','CHF','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDJPY','100961267',2, 'USDJPY','forex',False, 25000,'USD','JPY','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'AUDUSD','100961267',1, 'AUDUSD','forex',False, 25000,'AUD','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDCAD','100961267',1, 'USDCAD','forex',False, 25000,'USD','CAD','IDEALPRO','CASH', True)


adj_size(model_pos, ib_pos,'EURUSD','100984342',5, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100984342',5, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100984342',5, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100984342',5, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100984342',5, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100984342',5, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','100961267',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100961267',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100961267',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100961267',1, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100961267',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100961267',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)
