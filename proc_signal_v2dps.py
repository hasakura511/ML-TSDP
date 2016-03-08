import numpy as np
import pandas as pd
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order


import json
from pandas.io.json import json_normalize
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open
from c2api.get_exec import get_c2pos, get_exec_open as get_c2exec_open
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.order import adj_size
from time import gmtime, strftime, time, localtime, sleep
    
model_pos=get_dps_model_pos(['v2_AUDUSD'])
ib_pos=get_ibpos()

#proc_signal('EURUSD','100961226',1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
#proc_signal('GBPUSD','100962402',1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
#proc_signal('USDCHF','100962399',1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
#proc_signal('USDJPY','100962390',2, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
#proc_signal('USDCAD','100962769',1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'v2_AUDUSD','101008882',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'v2_AUDUSD','100962754',1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)

#proc_signal_dsp('v2_AUDUSD','101008882','AUDUSD','forex',True, 'AUD','USD','IDEALPRO','CASH', False)
#proc_signal_dsp('v2_AUDUSD','100962754','AUDUSD','forex',True, 'AUD','USD','IDEALPRO','CASH', False)


