import requests
from time import gmtime, strftime, time, localtime, sleep
import logging
import pandas as pd

data=pd.read_csv('./data/c2api/sigid.csv').iloc[-1]
sigid=int(data['sigid'])

def place_order(action, quant, sym, type, systemid, submit,apikey, parentsig=None):
    global sigid
    if submit == False:
        return 0;
    url = 'https://collective2.com/world/apiv2/submitSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    sigid=int(sigid)+1
    
    data = { 
    		"apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid, 
    		"signal":{ 
    	   		"action": action, 
    	   		"quant": quant, 
    	   		"symbol": sym, 
    	   		"typeofsymbol": type, 
    	   		"market": 1, 	#"limit": 31.05, 
    	   		"duration": "DAY", 
               "signalid": sigid,
               "conditionalupon": parentsig
    		} 
    	}
    logging.info( 'sigid is: ' + str( sigid ))
    dataf=pd.DataFrame([[sigid]], columns=['sigid'])
    dataf.to_csv('./data/c2api/sigid.csv')    
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
