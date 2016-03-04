import numpy as np
import pandas as pd
import time
import json
from pandas.io.json import json_normalize
from ibapi.get_exec import get_exec as get_ibexec
from c2api.get_exec import get_exec as get_c2exec

data=get_c2exec(100961267);
jsondata = json.loads(data)
dataSet=json_normalize(jsondata['response'])
dataSet.to_csv('./test.csv', index=False)

data=get_ibexec()
dataSet=pd.DataFrame(data)
dataSet.to_csv('./test2.csv', index=False)


