import numpy as np
import pandas as pd
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order

def adj_size(system, systemid, c2action, c2quant, c2sym, c2type, c2submit, ibaction, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    place_c2order(c2action, c2quant, c2sym, c2type, systemid, c2submit)
    place_iborder(ibaction, ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    

#adj_size('EURUSD','100961226','BTC', 1,'EURUSD','forex',True, 'SELL', 10000,'EUR','USD','IDEALPRO','CASH', False)
#adj_size('GBPUSD','100962402','BTC', 4,'GBPUSD','forex',True, 'SELL', 10000,'GBP','USD','IDEALPRO','CASH', False)
#adj_size('USDCHF','100962399','BTC', 4,'USDCHF','forex',True, 'SELL', 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size('USDJPY','100962390','BTC', 1,'USDJPY','forex',True, 'SELL', 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size('USDJPY','100962390','BTO', 1,'USDJPY','forex',True, 'SELL', 20000,'USD','JPY','IDEALPRO','CASH', False)
#adj_size('AUDUSD','100962756','STC', 1,'AUDUSD','forex',True, 'SELL', 10000,'AUD','USD','IDEALPRO','CASH', False)
#adj_size('USDCAD','100962769','STC', 1,'USDCAD','forex',True, 'SELL', 10000,'USD','CAD','IDEALPRO','CASH', False)

#adj_size('EURUSD','100961267','STC', 1,'EURUSD','forex',False, 'SELL', 25000,'EUR','USD','IDEALPRO','CASH', True)
#adj_size('GBPUSD','100961267','STC', 1,'GBPUSD','forex',False, 'BUY', 75000,'GBP','USD','IDEALPRO','CASH', True)
#adj_size('USDCHF','100961267','STC', 1,'USDCHF','forex',False, 'BUY', 75000,'USD','CHF','IDEALPRO','CASH', True)
#adj_size('USDJPY','100961267','STC', 1,'USDJPY','forex',False, 'SELL', 25000,'USD','JPY','IDEALPRO','CASH', True)
#adj_size('AUDUSD','100961267','STC', 1,'AUDUSD','forex',False, 'SELL', 25000,'AUD','USD','IDEALPRO','CASH', True)
#adj_size('USDCAD','100961267','STC', 1,'USDCAD','forex',False, 'SELL', 25000,'USD','CAD','IDEALPRO','CASH', True)

#adj_size('EURUSD','100984342','STC', 5,'EURUSD','forex',True, 'SELL', 10000,'EUR','USD','IDEALPRO','CASH', False)
#adj_size('GBPUSD','100984342','BTC', 5,'GBPUSD','forex',True, 'SELL', 10000,'GBP','USD','IDEALPRO','CASH', False)
#adj_size('USDCHF','100984342','STC', 5,'USDCHF','forex',True, 'SELL', 10000,'USD','CHF','IDEALPRO','CASH', False)
#adj_size('USDJPY','100984342','STC', 5,'USDJPY','forex',True, 'SELL', 10000,'USD','JPY','IDEALPRO','CASH', False)
#adj_size('AUDUSD','100984342','STC', 5,'AUDUSD','forex',True, 'SELL', 10000,'AUD','USD','IDEALPRO','CASH', False)
#adj_size('USDCAD','100984342','STC', 5,'USDCAD','forex',True, 'SELL', 10000,'USD','CAD','IDEALPRO','CASH', False)

#adj_size('EURUSD','100961267','BTC', 1,'EURUSD','forex',True, 'SELL', 10000,'EUR','USD','IDEALPRO','CASH', False)
#adj_size('GBPUSD','100961267','BTC', 2,'GBPUSD','forex',True, 'SELL', 10000,'GBP','USD','IDEALPRO','CASH', False)
#adj_size('USDCHF','100961267','BTC', 1,'USDCHF','forex',True, 'SELL', 10000,'USD','CHF','IDEALPRO','CASH', False)
#adj_size('USDJPY','100961267','STC', 1,'USDJPY','forex',True, 'SELL', 20000,'USD','JPY','IDEALPRO','CASH', False)
#adj_size('AUDUSD','100961267','STC', 1,'AUDUSD','forex',True, 'SELL', 10000,'AUD','USD','IDEALPRO','CASH', False)
#adj_size('USDCAD','100961267','STC', 1,'USDCAD','forex',True, 'SELL', 10000,'USD','CAD','IDEALPRO','CASH', False)