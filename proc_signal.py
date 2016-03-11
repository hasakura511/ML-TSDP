import numpy as np
import pandas as pd
import subprocess
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

model_pos=get_model_pos(['EURJPY','EURUSD','GBPUSD','USDCHF','USDJPY','AUDUSD','USDCAD'])
dps_model_pos=get_dps_model_pos(['v2_EURJPY','v2_EURUSD','v2_GBPUSD','v2_USDCHF','v2_USDJPY','v2_AUDUSD','v2_USDCAD'])

subprocess.call(['python', 'get_ibpos.py'])
#ib_pos=get_ibpos()
ib_pos=get_ibpos_from_csv()

data=pd.read_csv('./data/systems/system.csv')
data=data.reset_index()

for i in data.index:
        system=data.ix[i]
        model=model_pos
        if system['Version'] == 'v1':
                model=model_pos
        elif system['Version'] == 'v2':
                model=dps_model_pos'
        adj_size(model, ib_pos,system['System'],system['c2id'],system['c2api'],system['c2qty'],system['c2sym'],system['c2type'],system['c2submit'], system['ibqty'],system['ibsym'],system['ibcur'],system['ibexch'],system['ibtype'],system['ibsubmit'])
        time.sleep(1)
#subprocess.call(['python', 'proc_signal_v2.py'])
#subprocess.call(['python', 'get_ibpos.py'])
#subprocess.call(['python', 'proc_signal_v2dps.py'])
