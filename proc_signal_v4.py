import numpy as np
import pandas as pd
import subprocess
import time
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order
import json
from pandas.io.json import json_normalize
from ibapi.get_exec import get_ibpos, get_exec_open, get_ib_pos
from c2api.get_exec import get_c2pos, get_exec_open, retrieveSystemEquity, get_c2_pos
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading


logging.basicConfig(filename='/logs/proc_signal_v4.log',level=logging.DEBUG)
start_time = time.time()
debug=False


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

def start_systems(systemdata):
      threads = []        
      systemList=dict()
      #get c2 positions
      get_c2pos(systemdata)
      for i in systemdata.index:
          system=systemdata.ix[i]
          #print system, sys.argv[2]
          if len(sys.argv) < 2 or (len(sys.argv[2]) > 0 and sys.argv[2] == system['Name']):
              if systemList.has_key(system['Name']):
                  systemList[system['Name']]=systemList[system['Name']].append(system)
              else:
                  systemList[system['Name']]=pd.DataFrame()
                  systemList[system['Name']]=systemList[system['Name']].append(system)
              
      for systemname in systemList.keys():
           systems=systemList[systemname]
           systems['last_trade']=0
           systems['key']=systems['c2sym']
           systems=systems.set_index('key')
           sig_thread = threading.Thread(target=start_trade, args=[systems])
           sig_thread.daemon=True
           threads.append(sig_thread)
           sig_thread.start()
      [t.join() for t in threads]



def adj_size(model_pos, system, systemname, systemid, c2apikey, c2quant,\
                    c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype,\
                    ibsubmit, iblocalsym=''):
    system_pos=model_pos.loc[system]
   
    logging.info('==============')
    logging.info('Strategy:' + systemname)
    #logging.info('system_pos:' +str(system_pos))
    logging.info("  Signal Name: " + system)
    logging.info("  C2ID: " + systemid + "  C2Key: " + c2apikey)
    logging.info("  C2Sym: " + c2sym + " IBSym: " + ibsym)
    if c2submit == 'TRUE':
        c2submit=True
    elif c2submit == 'FALSE':
        c2submit=False
        
    if ibsubmit == 'TRUE':
        ibsubmit=True
    elif ibsubmit == 'FALSE':
        ibsubmit=False
    #print str(system_pos['action'])
    #print "c2: " 
    #print c2_pos
    if c2submit:
        c2_pos_qty=get_c2_pos(systemname, c2sym)           
        system_c2pos_qty=round(system_pos['action']) * c2quant
        logging.info( "system_c2_pos: " + str(system_c2pos_qty) )
        logging.info( "c2_pos: " + str(c2_pos_qty) )
        
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            isrev=False
            psigid=0
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                logging.info( 'BTC: ' + str(qty) )
                psigid=place_c2order('BTC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                isrev=True                
                c2quant = c2quant - qty
                
            if c2quant > 0:
                logging.info( 'BTO: ' + str(c2quant) )
                if isrev:
                    place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey, psigid)
                else:
                    place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            isrev=False
            psigid=0
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                logging.info( 'STC: ' + str(qty) )
                psigid=place_c2order('STC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                isrev=True 
                c2quant = c2quant - qty

            if c2quant > 0:
                logging.info( 'STO: ' + str(c2quant) )
                if isrev:
                    place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey, psigid)
                else:
                    place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
   
    if ibsubmit:
        ib_pos_qty=get_ib_pos(ibsym, ibcurrency)
        system_ibpos_qty=round(system_pos['action']) * ibquant
        
        logging.info( "system_ib_pos: " + str(system_ibpos_qty) )
        logging.info( "ib_pos: " + str(ib_pos_qty) )
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            logging.info( 'BUY: ' + str(ibquant) )
            place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            logging.info( 'SELL: ' + str(ibquant) )
            place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
    #
    #place_iborder(ibaction, ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);

#subprocess.call(['python', 'get_ibpos.py'])       
systemdata=pd.read_csv('./data/systems/system_'+sys.argv[2]+'.csv')
systemdata=systemdata.reset_index()
start_systems(systemdata)
get_executions(systemdata)
#subprocess.call(['python', 'get_ibpos.py'])
