import subprocess

with open('./data/futures.txt') as f:
    futures = f.read().splitlines()


version='v4'
f=open ('/logs/'  + 'createhtml.log','a')
print 'Starting createhtml: ' 
f.write('Starting createhtml: ' )

ferr=open ('/logs/createhtml_err.log','a')
ferr.write('Starting createhtml: ' )
subprocess.call(['python','create_signalPlotsFutures.py','1'], stdout=f, stderr=ferr)
for contract in futures:
	subprocess.call(['python','get_results.py','create',version+'_'+contract,'2'], stdout=f, stderr=ferr)
	
f.close()
ferr.close()
		