import requests

    
def get_working_signals(systemid):
    print "get_working_signals: " + systemid
    url = 'https://api.collective2.com/world/apiv3/retrieveSignalsWorking'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    return r.text


def cancel_signal(signalid, systemid):
    print "cancel_signal: systemid:" + systemid + ', signalid' + signalid
    
    url = 'https://api.collective2.com/world/apiv3/cancelSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid, 
    		"signalid": signalid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    print r.text