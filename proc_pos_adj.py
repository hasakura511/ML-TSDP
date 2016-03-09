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
    
model_pos=get_model_pos(['EURJPY','EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
ib_pos=get_ibpos()
#ib_pos=get_ibpos_from_csv()

adj_size(model_pos, ib_pos,'EURUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'EURUSD','forex',False, 25000,'EUR','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'GBPUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'GBPUSD','forex',False, 25000,'GBP','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDCHF','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCHF','forex',False, 25000,'USD','CHF','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDJPY','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDJPY','forex',False, 25000,'USD','JPY','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'AUDUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',False, 25000,'AUD','USD','IDEALPRO','CASH', True)
adj_size(model_pos, ib_pos,'USDCAD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCAD','forex',False, 25000,'USD','CAD','IDEALPRO','CASH', True)

adj_size(model_pos, ib_pos,'EURJPY','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'EURJPY','forex',True, 10000,'EUR','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'EURUSD','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",0, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",5, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','100961226',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','100962402',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','100962399',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','100962390',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDJPY','forex',True, 20000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','100962756',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','100962769',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
#adj_size(model_pos, ib_pos,'EURJPY','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'EURJPY','forex',True, 10000,'EUR','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','101092666',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",5, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

adj_size(model_pos, ib_pos,'EURUSD','101092600',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'EURUSD','forex',True, 10000,'EUR','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'EURJPY','101107744',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'EURJPY','forex',True, 10000,'EUR','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'GBPUSD','101092635',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'GBPUSD','forex',True, 10000,'GBP','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCHF','101092629',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'USDCHF','forex',True, 10000,'USD','CHF','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDJPY','101092615',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'USDJPY','forex',True, 10000,'USD','JPY','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'AUDUSD','101092650',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(model_pos, ib_pos,'USDCAD','101092609',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",10, 'USDCAD','forex',True, 10000,'USD','CAD','IDEALPRO','CASH', False)

