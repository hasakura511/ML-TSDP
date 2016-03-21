import numpy as np
import pandas as pd
import time

import json
from pandas.io.json import json_normalize
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ib_sym_pos
from c2api.get_exec import get_c2pos, get_exec_open as get_c2exec_open, reset_c2pos_cache
from seitoolz.signal import get_model_pos
from time import gmtime, strftime, time, localtime, sleep
    
def adj_size(model_pos, ib_pos, system, systemname, systemid, c2apikey, c2quant, c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype, ibsubmit):
    system_pos=model_pos.loc[system]
   
    print "system: " + system
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
        c2_pos=get_c2pos(systemid,c2sym,c2apikey,systemname).loc[c2sym]
        c2_pos_qty=int(c2_pos['quant_opened']) - int(c2_pos['quant_closed'])
        c2_pos_side=c2_pos['long_or_short']
        if c2_pos_side == 'short':
            c2_pos_qty=-c2_pos_qty
        
        system_c2pos_qty=round(system_pos['action']) * c2quant
        print "system_c2_pos: " + str(system_c2pos_qty)
        print "c2_pos: " + str(c2_pos_qty)
        
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'BTC: ' + str(qty)
                place_c2order('BTC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                reset_c2pos_cache(systemid)
                c2quant = c2quant - qty
                sleep(2)
            
            if c2quant > 0:
                print 'BTO: ' + str(c2quant)
                place_c2order('BTO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
                reset_c2pos_cache(systemid)
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print 'STC: ' + str(qty)
                place_c2order('STC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                reset_c2pos_cache(systemid)
                c2quant = c2quant - qty
                sleep(2)
            if c2quant > 0:
                print 'STO: ' + str(c2quant)
                place_c2order('STO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
                reset_c2pos_cache(systemid)
    if ibsubmit:
        ib_pos=get_ib_sym_pos(ib_pos, ibsym, ibcurrency) 
        ib_pos=ib_pos.loc[ibsym,ibcurrency]
        ib_pos_qty=ib_pos['qty']
        system_ibpos_qty=round(system_pos['action']) * ibquant
        print "system_ib_pos: " + str(system_ibpos_qty)
        print "ib_pos: " + str(ib_pos_qty)
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            print 'BUY: ' + str(ibquant)
            place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            print 'SELL: ' + str(ibquant)
            place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
    #
    #place_iborder(ibaction, ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
