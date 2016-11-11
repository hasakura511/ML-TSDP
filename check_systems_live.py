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
#from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ibpos_from_csv
from c2api.get_exec import get_exec_open, get_c2_list, get_c2livepos, retrieveSignalsWorking
#from seitoolz.signal import get_dps_model_pos, get_model_pos
#from seitoolz.order import adj_size
#from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
from datetime import datetime as dt

#logging.basicConfig(filename='/logs/check_systems.log',level=logging.DEBUG)
start_time = time.time()


if len(sys.argv)==1 or sys.argv[1]=='0':
    #systems = ['v4micro']
    systems = ['v4futures','v4mini','v4micro']
    logging.basicConfig(filename='C:/logs/c2.log',level=logging.DEBUG)
    logging.info('test')
    #savePath='D:/ML-TSDP/data/portfolio/'
    systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/'
    def place_order2(a,b,c,d,e,f,g):
        return "0"
else:
    systems = ['v4futures','v4mini','v4micro']
    logging.basicConfig(filename='/logs/c2.log',level=logging.DEBUG)
    #savePath='./data/portfolio/'
    systemPath =  './data/systems/'
    from c2api.place_order import place_order2

    
    
def reconcileWorkingSignals(sys, workingSignals, sym, sig, c2sig, qty, c2qty):
    errors=0
    print 'position mismatch: ', sym, 's:'+str(sig), 'c2s:'+str(c2sig), 'q:'+str(qty), 'c2q:'+str(c2qty),
    #position size adjustements
    if sig!=c2sig:
        #signal reversal there is no adjustment
        c2qty2=0
    else:
        #signal is the same there is an adjustment
        #c2qty2 existing,orders is adjustment
        c2qty2=c2qty
        
    if 'symbol' in workingSignals[sys] and sym in workingSignals[sys].symbol.values:
        orders = workingSignals[sys][workingSignals[sys].symbol==sym]
        orders.quant = orders.quant.astype(int)
        #Open orders
        if sig==1 and 'BTO' in orders.action.values:
            print 'working sig OK',
            #new open adjustment plus existing position equals qty in system file
            if orders[orders.action=='BTO'].quant.values[0] +c2qty2 == qty:
                print 'qty OK'
            else:
                print 'qty ERROR'
                errors+=1
        elif sig==-1 and 'STO' in orders.action.values:
            print 'working sig OK',
            if orders[orders.action=='STO'].quant.values[0] +c2qty2== qty:
                print 'qty OK'
            else:
                print 'qty ERROR'
                errors+=1
        #Close Orders where sig/qty = 0
        elif (sig==0 and ('STC' in orders.action.values or 'BTC' in orders.action.values))\
                or (qty==0 and ('STC' in orders.action.values or 'BTC' in orders.action.values)):
            print 'working sig OK',
            if orders.quant.values[0] == c2qty:
                print 'qty OK'
            else:
                print 'qty ERROR'
                errors+=1
        else:
            #Close Orders where qty adjustments
            if sig==c2sig:
                print 'working sig OK',
                #check for position adjustment
                if orders.action.values[0]=='BTC' and c2qty2-orders[orders.action=='BTC'].quant.values[0] == qty:
                    print 'qty OK'
                elif orders.action.values[0]=='STC' and c2qty2-orders[orders.action=='STC'].quant.values[0] == qty:
                    print 'qty OK'
                else:
                    #unknown scenario
                    print 'Wrong working qty found!'
                    errors+=1

            else:
                #unknown scenario
                print 'Wrong working signal found!'
                errors+=1
    else:
        print 'ERROR no working signals found!'
        errors+=1
    return errors
                
c2dict={}
workingSignals={}
futuresDict={}
for sys in systems:
    #subprocess.call(['python', 'get_ibpos.py'])       
    print sys, 'Getting c2 positions'
    systemdata=pd.read_csv(systemPath+'system_'+sys+'_live.csv')
    futuresDict[sys]=systemdata=systemdata.reset_index()
    futuresDict[sys].index=futuresDict[sys].c2sym
    #portfolio and equity
    c2list=get_c2_list(systemdata)
    c2systems=c2list.keys()
    for systemname in c2systems:
        (systemid, apikey)=c2list[systemname]
        c2dict[systemname]=get_c2livepos(systemid, apikey, systemname)
        response = json.loads(retrieveSignalsWorking(systemid, apikey))['response']
        if len(response)>0:
            workingSignals[systemname]= json_normalize(response)
        else:
            workingSignals[systemname]= response
    #trades
    #get_executions(systemdata)
    #subprocess.call(['python', 'get_ibpos.py'])

                    
for sys in c2dict.keys():
    print sys, 'Position Checking..'
    exitList=[]
    #sig reconciliation
    c2_count=0
    sys_count=0
    mismatch_count=0
    exit_count=0
    error_count=0
    c2dict[sys]['signal']=np.where(c2dict[sys]['long_or_short'].values=='long',1,-1)
    #check contracts in c2 file not in system file.
    for sym in c2dict[sys].index:
        c2_count+=1
        if sym in futuresDict[sys].index:

            c2sig = int(c2dict[sys].ix[sym].signal)
            sig=int(futuresDict[sys].ix[sym].signal)
            qty=int(futuresDict[sys].ix[sym].c2qty)
            c2qty=int(c2dict[sys].ix[sym].quant_opened)-int(c2dict[sys].ix[sym].quant_closed)
            if sig != c2sig or qty != c2qty:
                mismatch_count+=1
                error_count+=reconcileWorkingSignals(sys, workingSignals, sym, sig, c2sig, qty, c2qty)
                
        else:
            exit_count+=1
            c2sig = int(c2dict[sys].ix[sym].signal)
            c2qty=int(c2dict[sys].ix[sym].quant_opened)-int(c2dict[sys].ix[sym].quant_closed)
            symInfo=futuresDict[sys].ix[[x for x in futuresDict[sys].index if sym[:-2] in x][0]]
            if c2sig==1:
                action='STC'
            else:
                action='BTC'
                
            #place order to exit the contract.

            #old contract does not exist in system file so use the new contract c2id and api
            response = place_order2(action, c2qty, sym, symInfo.c2type, symInfo.c2id, True, symInfo.c2api)
            exitList.append(sym+' not in system file. exiting contract!!.. '+response)
    #check contracts in system file not in c2.
    for sym in futuresDict[sys].index:
        sig=int(futuresDict[sys].ix[sym].signal)
        qty=int(futuresDict[sys].ix[sym].c2qty)
        systemSym = (sig !=0 and qty !=0)
        if systemSym:
            sys_count+=1
        if sym not in c2dict[sys].index and systemSym:
            mismatch_count+=1
            error_count+=reconcileWorkingSignals(sys, workingSignals, sym, sig, 0, qty, 0)
            
    for e in exitList:
        print e
    print 'c2:'+str(c2_count)+' sys:'+str(sys_count)+' mismatch:'+str(mismatch_count)+' exit:'+str(exit_count)+' errors:'+str(error_count)
    print 'DONE!\n'

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()