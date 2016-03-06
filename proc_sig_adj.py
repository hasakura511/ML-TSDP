import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from time import gmtime, strftime, time, localtime
from c2api.sig_adj import get_working_signals, cancel_signal

def proc_sig_adj(systemid):
    data=get_working_signals(systemid);
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        for i in dataSet.index:
            row=dataSet.ix[i]
            cancel_signal(row['signal_id'], systemid)
     
proc_sig_adj('100961226')
proc_sig_adj('100962402')
proc_sig_adj('100962399')
proc_sig_adj('100962390')
proc_sig_adj('100962756')
proc_sig_adj('100962769')
proc_sig_adj('100961267')
proc_sig_adj('100984342')