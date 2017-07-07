#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:11:38 2017

@author: hidemiasakura
"""

import calendar
import os
from os.path import isfile
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



dbPath='./data/futures.sqlite3'
search_dir='./logs/'


class LogFiles(object):
    def __init__(self, filename):
        self.filename = filename

    def display_text_file(self):
        with open(self.filename) as fp:
            return fp.read()
            
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def MCdate():
    readConn = getBackendDB()
    timetables = pd.read_sql('select * from timetable', con=readConn, index_col='Desc')
    ttdates = timetables.drop(['Date','timestamp'],axis=1).columns.tolist()
    cutoff = datetime.time(17, 0, 0, 0)
    cutoff2 = datetime.time(23, 59, 59)
    eastern = timezone('US/Eastern')
    now = dt.now(get_localzone())
    now = now.astimezone(eastern)
    today = now.strftime("%Y%m%d")
    
    def guessMCdate():
        if now.time() > cutoff and now.time() < cutoff2:
            # M-SUN after cutoff, set next day
            next = now + datetime.timedelta(days=1)
            mcdate = next.strftime("%Y%m%d")
        else:
            # M-SUN before cutoff, keep same day
            mcdate = now.strftime("%Y%m%d")

        # overwrite weekends
        if now.weekday() == 4 and now.time() > cutoff and now.time() < cutoff2:
            # friday after cutoff so set to monday
            next = now + datetime.timedelta(days=3)
            mcdate = next.strftime("%Y%m%d")

        if now.weekday() == 5:
            # Saturday so set to monday
            next = now + datetime.timedelta(days=2)
            mcdate = next.strftime("%Y%m%d")

        if now.weekday() == 6:
            # Sunday so set to monday
            next = now + datetime.timedelta(days=1)
            mcdate = next.strftime("%Y%m%d")
        return mcdate
        
        
    if today in ttdates and ttdates.index(today)==0 and len(ttdates)>1:
        closes = pd.DataFrame(timetables[today].ix[[x for x in timetables.index if 'close' in x]].copy())
        lastclose=eastern.localize(max([x for x in pd.to_datetime(closes[today]) if x.day ==now.day]).to_pydatetime())
        if now>=lastclose :
            mcdate = ttdates[1]
        else:
            #now<lastclose
            mcdate =today
    elif len(ttdates)>0:
        next_ttdate = ttdates[-1]
        closes = pd.DataFrame(timetables[next_ttdate].ix[[x for x in timetables.index if 'close' in x]].copy())
        lastclose=eastern.localize(pd.to_datetime(closes[next_ttdate]).max().to_pydatetime())
        if now>=lastclose :
            #try last_ttdate
            #last_ttdate = ttdates[-1]
            #closes = pd.DataFrame(timetables[last_ttdate].ix[[x for x in timetables.index if 'close' in x]].copy())
            #lastclose = eastern.localize(pd.to_datetime(closes[last_ttdate]).max().to_pydatetime())
            #if (np.nan not in timetable[last_ttdate].tolist()) and now < lastclose:
            #    mcdate = last_ttdate
            #else:
            print('something wrong with timetable data. guessing next MCDATE')
            mcdate = guessMCdate()
        else:
            #now<lastclose
            mcdate =next_ttdate
    else:
        print('something wrong with timetable data. guessing next MCDATE')
        mcdate = guessMCdate()

    return mcdate

def getBackendDB():
    global dbPath
    readConn = sqlite3.connect(dbPath)
    return readConn

active_symbols_ib = {
    'v4futures': ['AUD', 'ZL', 'GBP', 'ZC', 'CAD', 'CL', 'EUR', 'EMD', 'ES', 'GF', 'ZF', 'GC', 'HG', 'HO', 'JPY',
                  'LE', 'HE', 'MXP', 'NZD', 'NG', 'NIY', 'NQ', 'PA', 'PL', 'RB', 'ZS', 'CHF', 'SI', 'ZM', 'ZT',
                  'ZN', 'ZB', 'ZW', 'YM'],
    'v4mini': ['ZC', 'CL', 'EUR', 'EMD', 'ES', 'HG', 'JPY', 'NG', 'ZM', 'ZT', 'ZN', 'ZW'],
    'v4micro': ['ZL', 'ES', 'HG', 'NG', 'ZN'],
    }
readConn = getBackendDB()
mcdate = MCdate()
eastern = timezone('US/Eastern')
utc = timezone('UTC')
now = dt.now(get_localzone()).astimezone(eastern)
now_str=now.strftime('%Y-%m-%d %I:%M:%S %p EST')
futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='IBsym')
timetables = pd.read_sql('select * from timetable', con=readConn, index_col='Desc').drop(['Date', 'timestamp'],
                                                                                         axis=1)

if mcdate in timetables:
    ttdate = mcdate
    # filter out any dates that have passed
    ttdates = [x for x in timetables.columns if int(x) >= int(mcdate)]
    timetables = timetables[ttdates]
else:
    # use old dates
    ttdate = timetables.columns[0]
#print mcdate, ttdate
timetableDF = pd.DataFrame()
#print timetables
for idx, [sym, value] in enumerate([x.split() for x in timetables.index]):
    idx2 = sym + ' ' + value
    timestamp = timetables.ix[idx2].ix[ttdate]
    #print sym, value, timestamp
    timetableDF.set_value(sym, value, timestamp)

#timetableDF.index.name = 'ibsym'
for sym in timetableDF.index:
    opentime = eastern.localize(dt.strptime(timetableDF.ix[sym].open, '%Y-%m-%d %H:%M'))
    closetime = eastern.localize(dt.strptime(timetableDF.ix[sym].close, '%Y-%m-%d %H:%M'))
    if now >= opentime and now < closetime:
        timetableDF.set_value(sym, 'immediate order type', 'Open: Market Order')
    else:
        # market is closed
        if now < opentime:
            nextopen = opentime.strftime('%A, %b %d %H:%M EST')
        elif len(timetables.drop(ttdate, axis=1).columns) > 0:
            next_ttdate = timetables.drop(ttdate, axis=1).columns[0]
            nextopen = timetables[next_ttdate].ix[sym + ' open']
            if not (nextopen is not None and nextopen is not np.nan):
                nextopen = 'Not Avalable'
            else:
                if now < eastern.localize(dt.strptime(nextopen, '%Y-%m-%d %H:%M')):
                    pass
                else:
                    nextopen = 'Not Available'
        else:
            nextopen = 'Not Available'
        timetableDF.set_value(sym, 'immediate order type', 'Closed: Market on Open ({})'.format(nextopen))

col_order2 = ['group', 'open', 'close', 'trigger']
col_order = ['group', 'immediate order type']
# timetableDF=timetableDF[col_order]
# print timetableDF
# ttDict={account:timetableDF.ix[active_symbols_ib[account]].to_html() for account in active_symbols_ib}
ttDict = {}
for account in active_symbols_ib:
    df = timetableDF.ix[active_symbols_ib[account]]
    df['group'] = futuresDict.ix[df.index].Group
    desc_list = futuresDict.ix[df.index].Desc.values
    df.index = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
    df.index = ['<a href="/static/images/v4_' + [futuresDict.index[i] for i, desc in enumerate(futuresDict.Desc) \
                  if re.sub(r'-[^-]*$', '', x) in desc][0] + '_BRANK.png" target="_blank">' + x + '</a>' for x in df.index]
    text='This table lets you know what order types will be used for '+account+' if immediate is entered now.<br><br>Server Time: '+now_str
    ttDict[account] = {
                        'text':text,
                        'html':df[col_order].sort_values(by='group').to_html(escape=False),
                        }
    if account == 'v4futures':
        text = 'Detailed Timetable.<br>All times in Eastern Standard Time.<br><br>Server Time: ' + now_str
        ttDict['info'] = {
                        'text':text,
                        'html':df[col_order2].sort_values(by='group').to_html(escape=False)
                        }
        
c2_equity = pd.read_sql('select * from (select * from c2_equity order by timestamp ASC) group by system', \
                        con=readConn, index_col='system')
c2_equity.updatedLastTimeET = pd.to_datetime(c2_equity.updatedLastTimeET)
