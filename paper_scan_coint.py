import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
import os
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
import adfapi.s103 as s103
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf
import re
import csv

#pairs=[['./data/from_IB/1 min_NZDJPY.csv', 'NZDJPY'],['./data/from_IB/1 min_CADJPY.csv', 'CADJPY']]
pairs=list()

#BTAPI
dataPath='./data/btapi/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('BTCUSD')

for file in files:
        if re.search(btcsearch, file):
                systemname=file
                #systemname = re.sub('BTCUSD_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
                reader = pd.read_csv(dataPath+file)
                row_count = reader.shape[0]
                if row_count > 3000:
                    pairs.append([dataPath+file, systemname, 1])
                else:
                    print "Skipping " + dataPath+file + " with " + str(row_count)
dataPath='./data/from_IB/'
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
btcsearch=re.compile('1 min')

for file in files:
        if re.search(btcsearch, file):
                systemname=file
                systemname = re.sub('1 min_','', systemname.rstrip())
                systemname = re.sub('.csv','', systemname.rstrip())
                reader = pd.read_csv(dataPath+file)
                row_count = reader.shape[0]
                if row_count > 3000:
                    pairs.append([dataPath+file, systemname, 1])
                else:
                    print "Skipping " + dataPath+file + " with " + str(row_count)
                    

def scan_coint(sysname, SST):
    global result
    global seen
    global pairs
    for [file1,sym1] in pairs:
        #print "sym: " + sym1
        #if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
        #    seen[sym1+sym2]=1
        #    seen[sym2+sym1]=1
        sig_thread = threading.Thread(target=proc_pair, args=[sym1])
        sig_thread.daemon=True
        threads.append(sig_thread)
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    result.to_csv('./data/adf/coint.csv',index=False)
    
def proc_pair(sym1):
    try:
        global result
        global SST
        global seen
        for [file2,sym2] in pairs:
            if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
                seen[sym1+sym2]=1
                seen[sym2+sym1]=1
                print "Sym1: " + sym1 + '(' + str(len(SST[sym1])) + ") Sym2: " + sym2 + '(' + str(len(SST[sym2])) + ')'
                (confidence,pv, hurst)=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
                print "Coint Confidence: " + str(confidence) + "%"
                if confidence >= 95:
                    rec=pd.DataFrame([[int(time.time()), sym1, sym2, confidence, pv, hurst]], columns=['Date','Symbol1','Symbol2','Confidence','Pv','Hurst']).iloc[-1]
                    result=result.append(rec)
    except Exception as e:
        print "Error getting coint" + str(e) + ' , ' +  str(sys.exc_info()[0])
        
sysname='ADFScan'
SST=seigraph.get_history(pairs, 'Close')

threads=[]
result=pd.DataFrame({}, columns=['Date','Symbol1','Symbol2','Confidence','Pv','Hurst'])
seen=dict()   
scan_coint(sysname, SST)
