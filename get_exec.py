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
    
get_c2trades(100961226, 'stratEURUSD') #stratEURUSD
get_c2trades(100962402, 'stratGBPUSD') #stratGBPUSD
get_c2trades(100962399, 'stratUSDCHF') #stratUSDCHF
get_c2trades(100962390, 'stratUSDJPY') #stratUSDJPY
get_c2trades(100962756, 'stratAUDUSD') #stratAUDUSD
get_c2trades(100962769, 'stratUSDCAD') #stratUSDCAD

get_c2trades(100961267, 'testStrat') #testStrat
get_c2trades(100984342, 'testStrat102') #testStrat102

get_ibtrades()


