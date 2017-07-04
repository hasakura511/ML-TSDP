#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:40:42 2017

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
import requests
from pandas.io.json import json_normalize

dbPath='./data/futures.sqlite3'
def getBackendDB():
    global dbPath
    readConn = sqlite3.connect(dbPath)
    return readConn

readConn=getBackendDB()
#accountvalue=pd.read_sql('select * from (select * from c2_equity where\
#                        system=\'{}\' order by timestamp ASC) group by Date'.format(account), con=readConn)
#accountvalue.index=pd.to_datetime(accountvalue.updatedLastTimeET)
apikey='O9WoxVj7DNXkpifMY_blqHpFg5cp3Fjqc7Aiu4KQjb8mXQlEVx'
systemid="110126294"

def requestAllTrades(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/requestAllTrades_overview'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    data = { 
    		"apikey":   str(apikey),    #"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid),
         "show_component_signals": 1,
          }

    params={}
    
    r=requests.post(url, params=params, json=data);
    #print r.text
    #logging.info(r.text)
    data=json.loads(r.text)
    #print systemid, apikey, data
    return data



def getAccountValues(refresh=False):
    readConn = getBackendDB()
    mcdate = MCdate()
    eastern = timezone('US/Eastern')
    utc = timezone('UTC')
    now = dt.now(get_localzone())
    now = now.astimezone(eastern)
    accountvalues = {}
    urpnls = {}

    if not debug and refresh:
        get_newtimetable()

    # ib
    ib_equity = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'NetLiquidation\'', \
                            con=readConn, index_col='Desc').iloc[0].to_frame().transpose()
    #print ib_equity
    timestamp = utc.localize(dt.utcfromtimestamp(ib_equity.timestamp)).astimezone(eastern)
    accountvalue = int(float(ib_equity.value[0]))
    accountvalues['v4futures'] = {
        'col1title': 'Account Value', 'col1value': accountvalue, \
        'col2title': 'Timestamp', 'col2value': timestamp.strftime('%Y-%m-%d %I:%M:%S %p EST')
    }

    ib_urpnl = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'UnrealizedPnL\' and currency=\'BASE\'', \
                           con=readConn, index_col='Desc').iloc[0].to_frame().transpose()
    timestamp = utc.localize(dt.utcfromtimestamp(ib_urpnl.timestamp)).astimezone(eastern)
    urpnl = int(float(ib_urpnl.value[0]))
    urpnls['v4futures'] = {
        'col1title': 'UnrealizedPnL', 'col1value': urpnl, \
        'col2title': 'Timestamp', 'col2value': timestamp.strftime('%Y-%m-%d %I:%M:%S %p EST')
    }

    # C2
    c2_equity = pd.read_sql('select * from (select * from c2_equity order by timestamp ASC) group by system', \
                            con=readConn, index_col='system')
    c2_equity.updatedLastTimeET = pd.to_datetime(c2_equity.updatedLastTimeET)

    #run checksystems and update values if last update >=1 day and not a T,W,TH,F,SA
    if (now-eastern.localize(c2_equity.updatedLastTimeET[0])).days>=1 and (now.weekday()>0 and now.weekday()<5):
        if not debug:
            run_checksystems()

    for system in c2_equity.drop(['v4futures'], axis=0).index:
        timestamp = c2_equity.ix[system].updatedLastTimeET
        accountvalue = int(c2_equity.ix[system].modelAccountValue)
        #urpnl = int(c2_equity.ix[system].equity)
        df=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by c2sym' % ('checkSystems_'+system),\
                con=readConn)
        urpnl=df.urpnl.astype(int).sum()
        urpnls[system] = {
            'col1title': 'UnrealizedPnL', 'col1value': urpnl, \
            'col2title': 'Timestamp', 'col2value': timestamp.strftime('%Y-%m-%d %I:%M:%S %p EST')
        }
        accountvalues[system] = {
            'col1title': 'Account Value', 'col1value': accountvalue, \
            'col2title': 'Timestamp', 'col2value': timestamp.strftime('%Y-%m-%d %I:%M:%S %p EST')
        }
    '''
    c2_v4micro = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4micro\')', con=readConn)

    c2_v4mini = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4mini\')', con=readConn)

    c2_v4futures = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4futures\')', con=readConn)
    '''

    #record = AccountData(value1=json.dumps(urpnls), value2=json.dumps(accountvalues),mcdate=mcdate,\
    #                                timestamp=getTimeStamp())
    #record.save()
    accountdata={'value1':json.dumps(urpnls), 'value2':json.dumps(accountvalues),'mcdate':mcdate,\
                                    'timestamp':getTimeStamp()}
    return accountdata

def get_status():
    eastern = timezone('US/Eastern')
    utc = timezone('UTC')
    pngPath='static/images/'
    readConn = getBackendDB()
    futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='C2sym')
    futuresDict2 = pd.read_sql('select * from Dictionary', con=readConn, index_col='CSIsym')

    #col_order = ['broker', 'account', 'contract', 'description', 'openedWhen', 'urpnl', \
    #             'broker_position', 'broker_qty', 'signal_check', 'qty_check', 'selection', 'order_type']
    #col_order_ib = ['desc', 'contracts', 'qty', 'price', 'value', 'avg_cost', 'unr_pnl', 'real_pnl', 'accountid',
    #             'currency', 'bet', 'ordertype', 'status', 'Date']
    col_order_status = ['selection', 'order_type', 'signal_check', 'qty_check']
    col_order_status_ib = ['bet', 'ordertype', 'status']
    col_order_pnl = ['contract', 'urpnl','broker_position', 'broker_qty', 'openedWhen']
    col_order_pnl_ib = ['contracts', 'unr_pnl', 'real_pnl', 'currency', 'value', 'qty', 'Date']
    orderstatus_dict={}
    #slippage_files={}
    accounts = ['v4micro', 'v4mini', 'v4futures']
    records_urls={
        'v4micro':'<a href="https://collective2.com/details/110125347" target="_blank">Collective 2: v4micro</a>',
        'v4mini':'<a href="https://collective2.com/details/110125449" target="_blank">Collective 2: v4mini</a>',
        'v4futures':'<a href="https://collective2.com/details/110126294" target="_blank">Collective 2: v4futures</a>'
        }
        
    def conv_sig(signals):
        sig = signals.copy()
        sig[sig == -1] = 'SHORT'
        sig[sig == 1] = 'LONG'
        sig[sig == 0] = 'NONE'
        return sig.values

    for account in accounts:
        orderstatus_dict[account]={}
        orderstatus_dict[account]['tab_list'] = ['Status', 'UnrealizedPNL', 'Slippage']
        #webSelection=pd.read_sql('select * from webSelection where timestamp=\
        #        (select max(timestamp) from webSelection)'
        #bet = eval(webSelection.selection[0])[account][0]
        df=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by c2sym' % ('checkSystems_'+account),\
                con=readConn)
        timestamp=utc.localize(dt.utcfromtimestamp(df.timestamp[0])).astimezone(eastern).strftime('%Y-%m-%d %I:%M:%S %p EST')
        df['system_signal'] = conv_sig(df['system_signal'])
        df['broker_position'] = conv_sig(df['broker_position'])
        desc_list=futuresDict.ix[[x[:-2] for x in df.contract.values]].Desc.values
        df['description']=[re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
        df=df.set_index(['description'])
        df.index=['<a href="/static/images/v4_' + [futuresDict2.index[i] for i, desc in enumerate(futuresDict2.Desc) \
                  if re.sub(r'-[^-]*$', '', x) in desc][0] + '_BRANK.png" target="_blank">' + x + '</a>' for x in df.index]
        orderstatus_dict[account]['title_txt']=account+' Order Status'
        orderstatus_dict[account]['status']=df[col_order_status].to_html(escape=False)
        orderstatus_dict[account]['status_text']='This table lets you know the status of your last bet/orders. For example, if your bet was correctly transmitted and received by your broker, it would say OK.<br><br>Last Update: '+timestamp
        orderstatus_dict[account]['pnl'] = df[col_order_pnl].to_html(escape=False)
        orderstatus_dict[account]['pnl_text']='This table displays the details of your open positions in your account portfolio.<br>Last Updated: '+str(timestamp)+'<br><br>For the closed trading record please go here: '+records_urls[account]
        csidate = pd.read_sql('select distinct csiDate from slippage where Name=\'{}\''.format(account), con=readConn).csiDate.tolist()[-1]
        orderstatus_dict[account]['slippage']=pngPath+account+'_c2_slippage_'+str(csidate)+'.png'
        orderstatus_dict[account]['slippage_text']='The slippage graph shows the datetime your last orders were entered and how much it differs from the official close price. With immediate orders slippage will show the net loss/gain you get from entering earlier than at the MOC<br><br>Last Update: '+str(timestamp)
        if account == "v4futures":
            df = pd.read_sql('select * from (select * from %s\
                    order by timestamp ASC) group by ibsym' % ('checkSystems_ib_' + account), \
                             con=readConn)
            df['contracts']=[df.ix[i].ibsym+df.ix[i].exp if df.ix[i].exp is not None else '' for i in df.index]
            timestamp = utc.localize(dt.utcfromtimestamp(df.timestamp[0])).astimezone(eastern).strftime(
                '%Y-%m-%d %I:%M:%S %p EST')
            desc_list = [futuresDict.reset_index().set_index('IBsym').ix[x].Desc for x in df.ibsym]
            df['desc'] = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
            df=df.set_index(['desc'])
            df.index = ['<a href="/static/images/v4_' + [futuresDict2.index[i] for i, desc in enumerate(futuresDict2.Desc) \
                         if re.sub(r'-[^-]*$', '', x) in desc][0] + '_BRANK.png" target="_blank">' + x + '</a>' for x in df.index]
            #orderstatus_dict[account+'_ib'] = df[col_order].to_html()
            orderstatus_dict[account]['title_txt'] = account + ' Order Status'
            orderstatus_dict[account]['status'] = df[col_order_status_ib].to_html(escape=False)
            orderstatus_dict[account]['status_text']='This table lets you know the status of your last bet/orders. For example, if your bet was correctly transmitted and received by your broker, it would say OK.<br><br>Last Update: '+timestamp
            orderstatus_dict[account]['pnl'] = df[col_order_pnl_ib].to_html(escape=False)
            orderstatus_dict[account]['pnl_text']='This table displays the details of your open positions in your account portfolio.<br>Last Updated: '+str(timestamp)+'<br><br>For the closed trading record please go here: '+records_urls[account]
            csidate = pd.read_sql('select distinct Date from ib_slippage where Name=\'{}\''.format(account),
                                  con=readConn).Date.tolist()[-1]
            #slippage_files[account+'_ib'] = str(csidate)
            orderstatus_dict[account]['slippage'] = pngPath+account + '_ib_slippage_' + str(csidate) + '.png'
            orderstatus_dict[account]['slippage_text']='The slippage graph shows the datetime your last orders were entered and how much it differs from the official close price. With immediate orders slippage will show the net loss/gain you get from entering earlier than at the MOC<br><br>Last Update: ' + str(
                timestamp)
    return orderstatus_dict