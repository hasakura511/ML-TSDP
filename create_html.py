import subprocess

futures =  [
				'AD',
				'BO',
				'BP',
				'C',
				'CC',
				'CD',
				'CL',
				'CT',
				'DX',
				'EC',
				'ED',
				'ES',
				'FC',
				'FV',
				'GC',
				'HG',
				'HO',
				'JY',
				'KC',
				'LB',
				'LC',
				'LN',
				'MD',
				'MP',
				'NG',
				'NQ',
				'NR',
				'O',
				'OJ',
				'PA',
				'PL',
				'RB',
				'RU',
				'S',
				'SB',
				'SI',
				'SM',
				'TU',
				'TY',
				'US',
				'W',
				'XX',
				'YM'
				]

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
		