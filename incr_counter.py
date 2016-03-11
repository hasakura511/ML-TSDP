import pandas as pd
import subprocess
data=pd.read_csv('./data/counter')
counter=data['counter']
counter=counter +1
data['counter']=counter
data=data.set_index('counter')
data.to_csv('./data/counter')
if int(counter) >= 7:
	subprocess.call(['python', 'get_ibpos.py'])
	subprocess.call(['python', 'proc_signal.py'])	
