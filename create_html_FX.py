import subprocess


with open('./data/currencies.txt') as f:
    currencyPairs = f.read().splitlines()

version='v4'
f=open ('/logs/'  + 'createhtml.log','a')
print 'Starting createhtml: ' 
f.write('Starting createhtml: ' )

ferr=open ('/logs/createhtml_err.log','a')
ferr.write('Starting createhtml: ' )
subprocess.call(['python','create_signalPlots_FX.py','1'], stdout=f, stderr=ferr)
for pair in currencyPairs:
	subprocess.call(['python','get_results.py','create',version+'_'+pair,'2'], stdout=f, stderr=ferr)
	
f.close()
ferr.close()