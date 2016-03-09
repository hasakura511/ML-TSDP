import requests
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import time


def get_exec(systemid, apikey):
    
    url = 'https://collective2.com/world/apiv3/requestTrades'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   apikey,#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    return r.text

def get_exec_open(systemid, apikey):
    
    url = 'https://collective2.com/world/apiv3/requestTradesOpen'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   apikey,#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    return r.text

def get_c2pos(systemid, c2sym,apikey):
    datestr=strftime("%Y%m%d", localtime())
    data=get_exec_open(systemid,apikey);
    
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        if c2sym not in dataSet['symbol'].values:
            dataSet=dataSet.append(pd.DataFrame([[c2sym,0,0,'none']],
                              columns=['symbol','quant_opened','quant_closed','long_or_short']))
        dataSet=dataSet.set_index(['symbol'])
        dataSet.to_csv('./data/portfolio/c2_' + c2sym + '_' + str(systemid)+ '_pos.csv')
        return dataSet
    else:
        dataSet=pd.DataFrame([[c2sym,0,0,'none']],
                              columns=['symbol','quant_opened','quant_closed','long_or_short'])
        dataSet=dataSet.set_index(['symbol'])
        dataSet.to_csv('./data/portfolio/c2_' + c2sym + '_' + str(systemid)+ '_pos.csv')
        return dataSet
        
#place_order('BTO','1','EURUSD','forex')
