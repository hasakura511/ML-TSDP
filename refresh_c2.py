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
from c2api.get_exec import get_c2equity
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.order import adj_size
from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
from datetime import datetime as dt

logging.basicConfig(filename='/logs/refresh_c2.log',level=logging.DEBUG)
start_time = time.time()

systems = ['v4futures','v4mini', 'v4micro']
for system in systems:
    #subprocess.call(['python', 'get_ibpos.py'])       
    systemdata=pd.read_csv('./data/systems/system_'+system+'.csv')
    systemdata=systemdata.reset_index()
    get_c2pos(systemdata)
    get_executions(systemdata)
    data=get_c2equity(systemdata)
    #subprocess.call(['python', 'get_ibpos.py'])


print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()