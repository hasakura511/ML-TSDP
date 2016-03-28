import requests
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import time
import logging

system_cache={}

def get_exec(systemid, apikey):
    global system_cache
    systemid=str(systemid)
    if systemid in system_cache:
        return system_cache[systemid]
    else:
        url = 'https://collective2.com/world/apiv3/requestTrades'
        
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        
        data = { 
        		"apikey":   apikey,#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
        		"systemid": systemid
        	}
        
        params={}
        
        r=requests.post(url, params=params, json=data);
        
        sleep(2)
        system_cache[systemid]=r.text

    return system_cache[systemid]

def get_exec_open(systemid, apikey):
    global system_cache
    systemid=str(systemid)
    if systemid in system_cache:
        return system_cache[systemid]
    else:
        url = 'https://collective2.com/world/apiv3/requestTradesOpen'
        
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        
        data = { 
        		"apikey":   apikey,#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
        		"systemid": systemid
        	}
        
        params={}
        
        r=requests.post(url, params=params, json=data);
        sleep(2)
        system_cache[systemid]=r.text
    
    return system_cache[systemid]

def get_c2pos(systemid, c2sym, apikey, systemname):
    datestr=strftime("%Y%m%d", localtime())
    data=get_exec_open(systemid,apikey);
    
    jsondata = json.loads(data)
    try:
        if len(jsondata['response']) > 0:
            dataSet=json_normalize(jsondata['response'])
            if c2sym not in dataSet['symbol'].values:
                dataSet=dataSet.append(pd.DataFrame([[c2sym,0,0,'none']],
                                  columns=['symbol','quant_opened','quant_closed','long_or_short']))
            dataSet=dataSet.set_index(['symbol'])
            dataSet.to_csv('./data/portfolio/c2_' + systemname + '_portfolio.csv')
            return dataSet
    except Exception as e:
        logging.error("get_c2pos", exc_info=True)
        
       
    dataSet=pd.DataFrame([[c2sym,0,0,'none']],
                          columns=['symbol','quant_opened','quant_closed','long_or_short'])
    dataSet=dataSet.set_index(['symbol'])
    dataSet.to_csv('./data/portfolio/c2_' + systemname + '_portfolio.csv')
    return dataSet
    


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
    return dataSet
    
def reset_c2pos_cache(systemid):
    global system_cache
    system_cache.pop(systemid, None)

#place_order('BTO','1','EURUSD','forex')
