import numpy as np
import pandas as pd
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order

def proc_signal(system, systemid, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    data = pd.read_csv('./data/signals/' + system + '.csv', index_col='Date')
    signals=data['Signal'];
  
    signal=signals[-1];
    print signal;
    if len(signals) > 1:
          if signals[-2] == signals[-1]:
              signal=0;
          if signals[-2] > signals[-1]:
              place_c2order('STC', c2quant, c2sym, c2type, systemid, c2submit)
              place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
          if signals[-2] < signals[-1]:
              place_c2order('BTC', c2quant, c2sym, c2type, systemid, c2submit)
              place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
     
    time.sleep( 10 )
    if signal == -1:
        place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit)
        place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    if signal == 1:
        place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit)
        place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    
    #os.remove('./data/signals/' + system + '.csv')
    

proc_signal('EURUSD','100961226',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
proc_signal('GBPUSD','100962402',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
proc_signal('USDCHF','100962399',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
proc_signal('USDJPY','100962390',2, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
proc_signal('AUDUSD','100962756',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
proc_signal('USDCAD','100962769',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

proc_signal('EURUSD','100961267',1, 'EURUSD','forex',False, 25000,'EUR','USD','IDEALPRO','CASH', True)
proc_signal('GBPUSD','100961267',1, 'GBPUSD','forex',False, 25000,'GBP','USD','IDEALPRO','CASH', True)
proc_signal('USDCHF','100961267',1, 'USDCHF','forex',False, 25000,'USD','CHF','IDEALPRO','CASH', True)
proc_signal('USDJPY','100961267',2, 'USDJPY','forex',False, 25000,'USD','JPY','IDEALPRO','CASH', True)
proc_signal('AUDUSD','100961267',1, 'AUDUSD','forex',False, 25000,'AUD','USD','IDEALPRO','CASH', True)
proc_signal('USDCAD','100961267',1, 'USDCAD','forex',False, 25000,'USD','CAD','IDEALPRO','CASH', True)


proc_signal('EURUSD','100984342',5, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
proc_signal('GBPUSD','100984342',5, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
proc_signal('USDCHF','100984342',5, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
proc_signal('USDJPY','100984342',5, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
proc_signal('AUDUSD','100984342',5, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
proc_signal('USDCAD','100984342',5, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

proc_signal('EURUSD','100961267',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
proc_signal('GBPUSD','100961267',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
proc_signal('USDCHF','100961267',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
proc_signal('USDJPY','100961267',1, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
proc_signal('AUDUSD','100961267',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
proc_signal('USDCAD','100961267',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)
