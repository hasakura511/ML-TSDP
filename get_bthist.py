import numpy as np
import pandas as pd
import time
import json
from time import gmtime, strftime, time, localtime

from pandas.io.json import json_normalize
from btapi.get_hist_coindesk import  get_hist_coindesk


def get_coindesk_hist():
    datestr=strftime("%Y%m%d", localtime())
    data=get_hist_coindesk();
    jsondata = json.loads(data)
    dataSet=json_normalize(jsondata['values'])
    dataSet.to_csv('./data/btapi/' + datestr + '_' + 'coindesk' + '_hist.csv', index=False)

get_coindesk_hist()

