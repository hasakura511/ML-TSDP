import pandas as pd
import subprocess
data=pd.read_csv('./data/counter')
counter=data['counter']
counter=counter +1
data['counter']=counter
data=data.set_index('counter')
data.to_csv('./data/counter')
if counter >= 7:
	subprocess.call(['python', 'proc_signal_v2dps.py'])	
