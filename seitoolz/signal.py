import numpy as np
import pandas as pd
import time
import os
import json
from pandas.io.json import json_normalize
from time import gmtime, strftime, time, localtime, sleep

def get_model_pos(systems):
    pos=pd.DataFrame({}, columns=['system','action','qty']);
    for system in systems:
        data = pd.read_csv('./data/signals/' + system + '.csv', index_col='Date')
        signals=data['Signal'];
        #safef=data['safef'];
       
        signal=signals[-1];
        qty=1
        pos=pos.append(pd.DataFrame([[system, signal, qty]], columns=['system','action','qty']))
        #pos=pos.append({'sym':system, 'action':signal, 'qty':qty}, ignore_index=True)
            
    pos=pos.set_index(['system'])
    
    pos.to_csv('./data/portfolio/model_pos.csv')
    return pos;

def get_dps_model_pos(systems):
    pos=pd.DataFrame({}, columns=['system','action','qty']);
    for system in systems:
        
        data = pd.read_csv('./data/signals/' + system + '.csv', index_col='dates')
        signals=data['signals'];
        safef=data['safef'];
        qty=safef[-1]*signals[-1]
        #signal=signals[-1];
        qty=round(qty)
       
        signal=qty
        
        pos=pos.append(pd.DataFrame([[system, signal, qty]], columns=['system','action','qty']))
        #pos=pos.append({'sym':system, 'action':signal, 'qty':qty}, ignore_index=True)
            
    pos=pos.set_index(['system'])
    
    pos.to_csv('./data/portfolio/dps_model_pos.csv')
    return pos;

def generate_model_pos(system):
    pos=pd.DataFrame({}, columns=['system','action','qty']);
    data = pd.read_csv('./data/signals/' + system + '.csv', index_col='dates')
    
    signals=data['signals'];
    safef=data['safef'];
    qty=safef[-1]*signals[-1]
    #signal=signals[-1];
    qty=round(qty)
    signal=qty
    
    pos=pos.append(pd.DataFrame([[system, signal, qty]], columns=['system','action','qty']))
    pos=pos.set_index(['system'])
    #pos.to_csv('./data/portfolio/gen_model_pos.csv')
    return pos

def generate_model_sig(system, date, action, qty):
    #pos=pd.DataFrame({}, columns=['system','action','qty']);
    pos=pd.DataFrame([[date, action, qty]], columns=['dates','signals','safef'])
    filename='./data/signals/' + system + '.csv'
    if os.path.isfile(filename):
        data=pd.read_csv(filename)
        pos=data.append(pos)
        pos=pos.set_index('dates')
        pos.to_csv('./data/signals/' + system + '.csv')
    else:
        pos=pos.set_index('dates')
        pos.to_csv('./data/signals/' + system + '.csv')

    return generate_model_pos(system)

def generate_model_manual(system, action, qty):
    pos=pd.DataFrame([[system, action, qty]], columns=['system','action','qty'])
    pos=pos.set_index(['system'])
    #pos.to_csv('./data/portfolio/gen_model_pos.csv')
    return pos