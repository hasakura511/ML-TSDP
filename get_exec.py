import numpy as np
import pandas as pd
import time
import json
from time import gmtime, strftime, time, localtime

from pandas.io.json import json_normalize
from ibapi.get_exec import get_exec as get_ibexec
from c2api.get_exec import get_exec as get_c2exec



def get_c2trades(systemid, name):
    datestr=strftime("%Y%m%d", localtime())
    data=get_c2exec(systemid);
    jsondata = json.loads(data)
    dataSet=json_normalize(jsondata['response'])
    dataSet.to_csv('./data/c2api/' + datestr + '_' + name + '_trades_' + str(systemid) + '.csv', index=False)

def get_ibtrades():
    datestr=strftime("%Y%m%d", localtime())
    data=get_ibexec()
    dataSet=pd.DataFrame(data)
    dataSet=dataSet.sort_values(by='times')
    dataSet.to_csv('./data/ibapi/' +  datestr + '_' + 'trades' + '.csv', index=False)

data=pd.read_csv('./data/systems/system.csv')
data=data.reset_index()

c2dict={}
for i in data.index:
        system=data.ix[i]
        c2dict[system['c2id']]=system['Name']

for c2id in c2dict:
	stratName=c2dict[c2id]
	get_c2trades(c2id, stratName)

get_ibtrades()


