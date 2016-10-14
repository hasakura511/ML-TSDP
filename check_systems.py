import requests
import os
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
#from ibapi.place_order import place_order as place_iborder
#from c2api.place_order import place_order as place_c2order
import json
from pandas.io.json import json_normalize
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ibpos_from_csv
from c2api.get_exec import get_exec_open, get_c2_list, get_c2livepos
from c2api.place_order import place_order, set_position
#from seitoolz.signal import get_dps_model_pos, get_model_pos
#from seitoolz.order import adj_size
#from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
from datetime import datetime as dt

logging.basicConfig(filename='/logs/check_systems.log',level=logging.DEBUG)
start_time = time.time()
systems = ['v4futures','v4mini','v4micro']

if len(sys.argv)==1:
    savePath='D:/ML-TSDP/data/portfolio/'
    systemPath = 'D:/ML-TSDP/data/systems/'
else:
    savePath='./data/portfolio/'
    systemPath =  './data/systems/'

c2dict={}
futuresDict={}
for sys in systems:
    #subprocess.call(['python', 'get_ibpos.py'])       
    systemdata=pd.read_csv(systemPath+'system_'+sys+'.csv')
    futuresDict[sys]=systemdata=systemdata.reset_index()
    futuresDict[sys].index=futuresDict[sys].c2sym
    #portfolio and equity
    c2list=get_c2_list(systemdata)
    systems=c2list.keys()
    for systemname in systems:
        (systemid, apikey)=c2list[systemname]
        c2dict[systemname]=get_c2livepos(systemid, apikey, systemname)
        
    #trades
    #get_executions(systemdata)
    #subprocess.call(['python', 'get_ibpos.py'])

for sys in c2dict.keys():
    print sys, 'Position Checking..'
    exitList=[]
    #sig reconciliation
    count=0
    c2dict[sys]['signal']=np.where(c2dict[sys]['long_or_short'].values=='long',1,-1)
    for sym in c2dict[sys].index:
        count+=1
        if sym in futuresDict[sys].index:
            c2sig = int(c2dict[sys].ix[sym].signal)
            sig=int(futuresDict[sys].ix[sym].signal)
            qty=int(futuresDict[sys].ix[sym].c2qty)
            c2qty=int(c2dict[sys].ix[sym].quant_opened)-int(c2dict[sys].ix[sym].quant_closed)
            if sig != c2sig or qty != c2qty:
                print 'position mismatch: ', sym, 's:'+str(sig), 'c2s:'+str(c2sig), 'q:'+str(qty), 'c2q:'+str(c2qty)
        else:
            exitList.append(sym+' not in system file. exit contract!!..')
            #place order to exit the contract.
            positions = {
                "symbol" : sym,
                "typeofsymbol" : "future",
                "quant" : 0
                }
            #old contract does not exist in system file so use the new contract c2id and api
            symInfo=futuresDict[sys].ix[[x for x in futuresDict[sys].index if sym[:-2] in x][0]]
            set_position(positions, symInfo.c2id, True, symInfo.c2api)
            
            
    for e in exitList:
        print e
    print count,'DONE!'

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()