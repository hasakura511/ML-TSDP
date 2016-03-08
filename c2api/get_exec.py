import requests

def get_exec(systemid):
    
    url = 'https://collective2.com/world/apiv3/requestTrades'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    return r.text

def get_exec_open(systemid):
    
    url = 'https://collective2.com/world/apiv3/requestTradesOpen'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    return r.text

#place_order('BTO','1','EURUSD','forex')
