import numpy as np
import pandas as pd
import time

import json
from pandas.io.json import json_normalize
from ibapi.place_order import place_order as place_iborder
from c2api.place_order import place_order as place_c2order
from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open
from c2api.get_exec import get_c2pos, get_exec_open as get_c2exec_open
from seitoolz.signal import get_model_pos
from seitoolz.order import adj_size
from time import gmtime, strftime, time, localtime, sleep
    
ib_pos=get_ibpos()
c2_pos=get_c2pos()

