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

dps_model_pos=get_dps_model_pos(['v2_AUDUSD'])
adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','101008882',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','100962754',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
#adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','100984342',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
#adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','100961267',"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w",1, 'AUDUSD','forex',False, 25000,'AUD','USD','IDEALPRO','CASH', True)
adj_size(dps_model_pos, ib_pos,'v2_AUDUSD','101092852',"bAOXHw3zGAxwXChl0YC_jv8joQBPFKR60AslojabW_lnYvTut9",1, 'AUDUSD','forex',True, 10000,'AUD','USD','IDEALPRO','CASH', False)
