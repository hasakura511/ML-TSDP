import numpy as np
import pandas as pd
import time

import json
from pandas.io.json import json_normalize
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ibpos_from_csv
from c2api.get_exec import get_c2pos, get_exec_open as get_c2exec_open
from seitoolz.signal import get_model_pos
from seitoolz.order import adj_size
from time import gmtime, strftime, time, localtime, sleep
    
model_pos=get_model_pos(['EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
ib_pos=get_ibpos()
#ib_pos=get_ibpos_from_csv()

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
#adj_size(model_pos, ib_pos,'AUDUSD','100961267',1, 'AUDUSD','forex',False, 25000,'AUD','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDCAD','100961267',1, 'USDCAD','forex',False, 25000,'USD','CAD','IDEALPRO','CASH', True)

adj_size(model_pos, ib_pos,'EURUSD','100984342',5, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100984342',5, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100984342',5, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100984342',5, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
#adj_size(model_pos, ib_pos,'AUDUSD','100984342',5, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100984342',5, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','100961267',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100961267',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100961267',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100961267',1, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100961267',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100961267',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

