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
from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading



logging.basicConfig(filename='/logs/proc_signal_v4.log',level=logging.DEBUG)
start_time = time.time()
debug=False
signalPath = './data/signals/'
dataPath = './data/from_IB/'

if len(sys.argv) > 1 and sys.argv[1] == '1':
    debug=True

def get_timestamp():
	timestamp = int(time.time())
	return timestamp
    
def get_models(systems):
    dpsList=dict()
    for i in systems.index:
        system=systems.ix[i]
        dpsList[system['System']]=1
    dps_model_pos=get_dps_model_pos(dpsList.keys())    
    return dps_model_pos
    
def start_trade(systems): 
        global debug
        if debug:
           print "Starting " + str(systems.iloc[0]['Name'])
           logging.info("Starting " + str(systems.iloc[0]['Name']))
        try:
            #model=get_models(systems)
            model = pd.concat([systems.System, systems.signal, systems.c2qty], axis=1)
            model.columns = ['system','action','qty']
            model=model.set_index(['system'])
            
            symbols=systems['c2sym'].values
            for symbol in symbols:
              system=systems.loc[symbol].copy()
              symbol=system['ibsym']
              if system['ibtype'] == 'CASH':
                    symbol = str(system['ibsym']) + str(system['ibcur'])
              
              #feed_dict=bars.get_bidask_list()
              if system['ibtype'] != 'BITCOIN':
                #and get_timestamp() - int(system['last_trade']) > int(system['trade_freq']):
                if system['c2submit'] or system['ibsubmit']:
                    adj_size(model, system['System'], system['Name'], 
                             str(int(system['c2id'])),system['c2api'],
                             system['c2qty'],system['c2sym'],system['c2type'],system['c2submit'], 
                             system['ibqty'],system['ibsym'],system['ibcur'],
                             system['ibexch'],system['ibtype'],system['ibsubmit'],
                             system['iblocalsym'])
              #time.sleep(30)
        except Exception as e:
            logging.error("something bad happened", exc_info=True)

def refresh_c2(systemdata):
      #get c2 positions
      get_c2pos(systemdata)

#subprocess.call(['python', 'get_ibpos.py'])       
systemdata=pd.read_csv('./data/systems/system_'+sys.argv[2]+'.csv')
systemdata=systemdata.reset_index()
refresh_c2(systemdata)
get_executions(systemdata)
#subprocess.call(['python', 'get_ibpos.py'])