
import numpy as np
import pandas as pd
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order

def proc_signal(system, systemid, c2quant, c2sym, c2type, ibquant, ibsym, ibcurrency, ibexch, ibtype):
    data = pd.read_csv('./data/signals/' + system + '.csv', index_col='Date')
    signals=data['Signal'];
  
    signal=signals[-1];
    print signal;
    if len(signals) > 1:
          if signals[-2] == signals[-1]:
              signal=0;
          if signals[-2] > signals[-1]:
              place_c2order('STC', c2quant, c2sym, c2type, systemid)
              place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch);
          if signals[-2] < signals[-1]:
              place_c2order('BTC', c2quant, c2sym, c2type, systemid)
              place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch);
     
    if signal == -1:
        place_c2order('STO', c2quant, c2sym, c2type, systemid)
        place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch);
    if signal == 1:
        place_c2order('BTO', c2quant, c2sym, c2type, systemid)
        place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch);
    
    #os.remove('./data/signals/' + system + '.csv')
    

proc_signal('EURUSD','100961267',1, 'EURUSD','forex',100000,'EUR','USD','IDEALPRO','CASH')
proc_signal('GBPUSD','100961267',1, 'GBPUSD','forex',100000,'GBP','USD','IDEALPRO','CASH')
proc_signal('USDCHF','100961267',1, 'USDCHF','forex',100000,'USD','CHF','IDEALPRO','CASH')
proc_signal('USDJPY','100961267',2, 'USDJPY','forex',200000,'USD','JPY','IDEALPRO','CASH')
proc_signal('AUDUSD','100961267',1, 'AUDUSD','forex',100000,'AUD','USD','IDEALPRO','CASH')
proc_signal('USDCAD','100961267',1, 'USDCAD','forex',100000,'USD','CAD','IDEALPRO','CASH')

proc_signal('EURUSD','100961226',1, 'EURUSD','forex',100000,'EUR','USD','IDEALPRO','CASH')
proc_signal('GBPUSD','100962402',1, 'GBPUSD','forex',100000,'GBP','USD','IDEALPRO','CASH')
proc_signal('USDCHF','100962399',1, 'USDCHF','forex',100000,'USD','CHF','IDEALPRO','CASH')
proc_signal('USDJPY','100962390',2, 'USDJPY','forex',200000,'USD','JPY','IDEALPRO','CASH')
proc_signal('AUDUSD','100962756',1, 'AUDUSD','forex',100000,'AUD','USD','IDEALPRO','CASH')
proc_signal('USDCAD','100962769',1, 'USDCAD','forex',100000,'USD','CAD','IDEALPRO','CASH')
