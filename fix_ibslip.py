# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import calendar
import os
import re
import time
import math
import json
import datetime
import numpy as np
from datetime import datetime as dt
from pytz import timezone
from tzlocal import get_localzone
import sqlite3
import pandas as pd
from os.path import isfile, join
from os import listdir

dbPath='./data/futures.sqlite3'
filePath='./data/portfolio/'
conn = sqlite3.connect(dbPath)
files=listdir(filePath)
slippage_files= [x for x in files if 'v4futures_ib_slippage_report_' in x]

for i,sf in enumerate(slippage_files):
    slippage_df=pd.read_csv(filePath+sf)
    print i, sf
    cols=slippage_df.columns.tolist()
    if 'timedelta' in slippage_df.columns:
        cols[cols.index('timedelta')]='delta'
        slippage_df.columns=cols
    slippage_df['Name']='v4futures'
    slippage_df['Date']=slippage_df.closetime[0][:10].replace('-','')
    slippage_df['timestamp']=int(calendar.timegm(pd.to_datetime(slippage_df.closetime)[0].to_datetime().utctimetuple()))
    if i==0:
        slippage_df.to_sql(name='ib_slippage',con=conn,if_exists='replace', index=False)
    else:
        slippage_df.to_sql(name='ib_slippage', con=conn,if_exists='append', index=False)
