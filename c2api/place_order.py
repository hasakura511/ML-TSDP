import requests

def place_order(action, quant, sym, type, systemid):
	url = 'https://collective2.com/world/apiv2/submitSignal'
	headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

	data = { 
		"apikey":   "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
		"systemid": systemid, 
		"signal":{ 
	   		"action": action, 
	   		"quant": quant, 
	   		"symbol": sym, 
	   		"typeofsymbol": type, 
	   		"market": 1, 	#"limit": 31.05, 
	   		"duration": "DAY", 
	   				#"signalid": 1011 
		} 
	}

	params={}

	r=requests.post(url, params=params, json=data);
	print r.text

#place_order('BTO','1','EURUSD','forex')
