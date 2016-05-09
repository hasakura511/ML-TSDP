import subprocess

with open('./data/currencies2.txt') as f:
    currencyPairs = f.read().splitlines()


f=open ('/logs/'  + 'getcurrencies.log','a')
print 'Starting getcurrencies: ' 
f.write('Starting getcurrencies: ' )

ferr=open ('/logs/getcurrencies_err.log','a')
ferr.write('Starting getcurrencies: ' )
#subprocess.call(['python','create_signalPlotsFutures.py','1'], stdout=f, stderr=ferr)
for pair in currencyPairs:
	subprocess.call(['python','get_hist.py',pair,'1h','5000'], stdout=f, stderr=ferr)
	
f.close()
ferr.close()
		