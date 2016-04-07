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
from time import gmtime, strftime, localtime, sleep
import logging

logging.basicConfig(filename='/logs/proc_signal.log',level=logging.DEBUG)

currencyList=dict()
v1sList=dict()
dpsList=dict()
systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
for i in systemdata.index:
    system=systemdata.ix[i]
    if system['ibtype'] != 'BITCOIN':
      currencyList[system['ibsym']+system['ibcur']]=1
      if system['Version'] == 'v1':
          v1sList[system['System']]=1
      else:
          dpsList[system['System']]=1

currencyPairs = currencyList.keys()
model_pos=get_model_pos(v1sList.keys())
dps_model_pos=get_dps_model_pos(dpsList.keys())

subprocess.call(['python', 'get_ibpos.py'])
#ib_pos=get_ibpos()
ib_pos=get_ibpos_from_csv()

for i in systemdata.index:
    try:
       system=systemdata.ix[i]
       model=model_pos
       if system['ibtype'] != 'BITCOIN':
            if system['Version'] == 'v1':
                    model=model_pos
            else:
                    model=dps_model_pos
            if system['c2submit'] or system['ibsubmit']:
                adj_size(model, system['System'], system['Name'], str(system['c2id']),system['c2api'],system['c2qty'],
                         system['c2sym'],system['c2type'],system['c2submit'], system['ibqty'],system['ibsym'],system['ibcur'],
                         system['ibexch'],system['ibtype'],system['ibsubmit'])
                #time.sleep(1)
    except Exception as e:
        logging.error("something bad happened", exc_info=True)

subprocess.call(['python', 'get_ibpos.py'])
#subprocess.call(['python', 'proc_signal_v2.py'])
#subprocess.call(['python', 'get_ibpos.py'])
#subprocess.call(['python', 'proc_signal_v2dps.py'])
