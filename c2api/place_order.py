import requests
from time import gmtime, strftime, time, localtime, sleep
import logging

sigid=1000

def place_order(action, quant, sym, type, systemid, submit,apikey, parentsig=None):
    if submit == False:
        return 0;
    url = 'https://collective2.com/world/apiv2/submitSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    sigid=sigid+1
    data = { 
    		"apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid, 
    		"signal":{ 
    	   		"action": action, 
    	   		"quant": quant, 
    	   		"symbol": sym, 
    	   		"typeofsymbol": type, 
    	   		"market": 1, 	#"limit": 31.05, 
    	   		"duration": "GTC", 
               "signalid": sigid,
               "conditionalupon": parentsig
    		} 
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    #sleep(2)
    #print r.text
    logging.info( str(r.text)  )
    return sigid

def set_position(positions, systemid, submit, apikey):
    if submit == False:
        return 0;
    url = 'https://api.collective2.com/world/apiv3/setDesiredPositions'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
          "apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
          "systemid": systemid, 
          "positions": positions
          #{
          #   "symbol" : "MSFT",
          #   "typeofsymbol" : "stock",
          #   "quant" : -30
          #},
          # {
          #    "symbol" : "@ESH6",
          #    "typeofsymbol" : "future",
          #   "quant" : 1
          # }
          #
          #
    }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    #print r.text
    logging.info( str(r.text)  )
#place_order('BTO','1','EURUSD','forex')
