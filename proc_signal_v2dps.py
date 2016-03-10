import numpy as np
import pandas as pd
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order


import json
from pandas.io.json import json_normalize
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ibpos_from_csv
from c2api.get_exec import get_c2pos, get_exec_open as get_c2exec_open
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.order import adj_size
from time import gmtime, strftime, time, localtime, sleep
    
#ib_pos=get_ibpos()
ib_pos=get_ibpos_from_csv()

dps_model_pos=get_dps_model_pos(['v2_EURJPY','v2_EURUSD','v2_GBPUSD','v2_USDCHF','v2_USDJPY','v2_AUDUSD','v2_USDCAD'])

adj_size(dps_model_pos, ib_pos,'v2_EURUSD','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_EURJPY','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'EURJPY','forex',True, 10000,'EUR','JPY','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_GBPUSD','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDCHF','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDJPY','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDCAD','101138810',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(dps_model_pos, ib_pos,'v2_EURUSD','101138698',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_EURJPY','101138480',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'EURJPY','forex',True, 10000,'EUR','JPY','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_GBPUSD','101138756',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDCHF','101138777',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDJPY','101138782',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','101092852',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_USDCAD','101138787',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

