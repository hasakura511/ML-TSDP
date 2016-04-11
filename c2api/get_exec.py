import requests
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import time
import logging
import os

def get_exec(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/requestTrades'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    data = { 
    		"apikey":   str(apikey),#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid)
          }
    params={}
    r=requests.post(url, params=params, json=data);
    print r.text
    logging.info(r.text)
    return r.text

def get_exec_open(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/requestTradesOpen'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    data = { 
    		"apikey":   str(apikey),    #"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid)
          }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    print r.text
    logging.info(r.text)
    return r.text

def get_c2pos():
    c2list=get_c2_list()
    systems=c2list.keys()
    for systemname in systems:
        (systemid, apikey)=c2list[systemname]
        c2list[systemname]=get_c2livepos(systemid, apikey, systemname)
    return c2list
    


def get_c2livepos(systemid, apikey, systemname):
    data=get_exec_open(systemid,apikey);
    
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        dataSet=dataSet.set_index(['symbol'])
        dataSet.to_csv('./data/portfolio/c2_' + systemname + '_portfolio.csv')
        return dataSet
        
def get_c2pos_from_csv(systemname):
    dataSet = pd.read_csv('./data/portfolio/c2_' + systemname + '_portfolio.csv', index_col='symbol')
    


def get_c2_portfolio(systemname):
    filename='./data/portfolio/c2_' + systemname + '_portfolio.csv'
     
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
        return dataSet
        
    else:
        dataSet=pd.DataFrame({},columns=['symbol','open_or_closed','long_or_short','quant_opened', 'quant_closed', \
                     'opening_price_VWAP', 'closing_price_VWAP', 'PL', 'PurePL', 'commission', \
                     'trade_id','closeVWAP_timestamp','closedWhen','closedWhenUnixTimeStamp', \
                     'expir','instrument',\
                     'markToMarket_time','openVWAP_timestamp','openedWhen','qty'])
        dataSet = dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        return dataSet

    
def get_c2_pos(systemname, c2sym):
    
    portfolio_data=get_c2_portfolio(systemname)
    portfolio_data=portfolio_data.reset_index()
    sym_cur=c2sym
    portfolio_data=portfolio_data.set_index('symbol')
    if sym_cur not in portfolio_data.index.values:
       return 0
    else:
        c2_pos=portfolio_data.loc[sym_cur]
        c2_pos_qty=float(c2_pos['quant_opened']) - float(c2_pos['quant_closed'])
        c2_pos_side=str(c2_pos['long_or_short'])
        if c2_pos_side == 'short':
            c2_pos_qty=-abs(c2_pos_qty)
        return c2_pos_qty
#place_order('BTO','1','EURUSD','forex')
def get_c2_list():
    dpsList=dict()
    
    systemdata=pd.read_csv('./data/systems/system.csv')
    systemdata=systemdata.reset_index()
    for i in systemdata.index:
        system=systemdata.ix[i]
        if system['c2submit']:
            dpsList[system['Name']]=[system['c2id'], system['c2api']]
        
    return dpsList
    