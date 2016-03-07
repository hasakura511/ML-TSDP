import numpy as np
import pandas as pd
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order

def proc_signal_dsp(system, systemid,  c2sym, c2type, c2submit, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    data = pd.read_csv('./data/signals/' + system + '.csv', index_col='dates')
    signals=data['signals'];
    safef=data['safef'];
    if safef[-1]*signals[-1] == 0:
        qty=safef[-2]*signals[-2]
     else:
        qty=safef[-1]*signals[-1]-safef[-2]*signals[-2]
    #signal=signals[-1];
    qty=round(qty)
    c2quant=qty
    ibquant=qty * 10000
  
    print qty;
    if safef[-1]*signals[-1]== 0:
      if safef[-2]*signals[-2] > 0:
          place_c2order('STC', c2quant, c2sym, c2type, systemid, c2submit)
          place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
      if safef[-2]*signals[-2] < 0:
          place_c2order('BTC', c2quant, c2sym, c2type, systemid, c2submit)
          place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);       
      if safef[-2]*signals[-2] == 0:
        pass
      #if len(signals) > 1:
      #        if signals[-2] == signals[-1]:
      #            signal=0;

    else:
    　time.sleep( 10 )
    　if qty<0:
        place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit)
        place_iborder("SELL", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    　if qty>0:
        place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit)
        place_iborder("BUY", ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
        
        #os.remove('./data/signals/' + system + '.csv')
    

#proc_signal('EURUSD','100961226',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
#proc_signal('GBPUSD','100962402',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
#proc_signal('USDCHF','100962399',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
#proc_signal('USDJPY','100962390',2, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
#proc_signal('USDCAD','100962769',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)
proc_signal_dsp('v2_AUDUSD','101008882','AUDUSD','forex',True, 'AUD','USD','IDEALPRO','CASH', False)
proc_signal_dsp('v2_AUDUSD','100962754','AUDUSD','forex',True, 'AUD','USD','IDEALPRO','CASH', False)


