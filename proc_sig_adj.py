import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import time
from c2api.sig_adj import get_working_signals, cancel_signal
seen=dict()

def proc_sig_adj(systemid,apikey):
    data=get_working_signals(systemid,apikey);
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        for i in dataSet.index:
            row=dataSet.ix[i]
            cancel_signal(row['signal_id'], systemid,apikey)
            #time.sleep(1)

data=pd.read_csv('./data/systems/system_backup.csv')
data=data.reset_index()

for i in data.index:
        system=data.ix[i]
        if system['c2submit'] and not seen.has_key(str(system['c2id'])):
            try:
                proc_sig_adj(str(system['c2id']),system['c2api'])
                seen[str(system['c2id'])]=1
            except Exception as e:
                print 'Error on', system['c2id']
