from django import forms
import time
import json
import datetime
import numpy as np
from datetime import datetime as dt
from pytz import timezone
from tzlocal import get_localzone
import sqlite3
import pandas as pd
from .models import MetaData, AccountData
from .start_moc import start_moc
import calendar
import os

def getBackendDB():
    dbPath = '/ML-TSDP/data/futures.sqlite3'
    readConn = sqlite3.connect(dbPath)
    return readConn

def get_logfiles(search_string=''):
    search_dir = "/logs/"
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in files if search_string in f]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    return files

class LoginForm(forms.Form):
    username = forms.CharField(label='User Name', max_length=64)
    password = forms.CharField(widget=forms.PasswordInput())

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
        lastclose=eastern.localize(pd.to_datetime(closes[today]).max().to_pydatetime())
        if now>=lastclose :
            mcdate = ttdates[1]
        else:
            #now<lastclose
            mcdate =today
    elif len(ttdates)>0:
        lastttdate = ttdates[-1]
        closes = pd.DataFrame(timetables[lastttdate].ix[[x for x in timetables.index if 'close' in x]].copy())
        lastclose=eastern.localize(pd.to_datetime(closes[lastttdate]).max().to_pydatetime())
        if now>=lastclose :
            print('something wrong with timetable data. guessing next MCDATE')
            mcdate = guessMCdate()
        else:
            #now<lastclose
            mcdate =lastttdate
    else:
        print('something wrong with timetable data. guessing next MCDATE')
        mcdate = guessMCdate()

    return mcdate

def getTimeStamp():
    timestamp = int(calendar.timegm(dt.utcnow().utctimetuple()))
    return timestamp

def get_futures_dictionary():
    readConn = getBackendDB()
    futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='CSIsym')
    groupdict = {group: {sym: futuresDict.ix[sym].to_dict() for sym in futuresDict.index if\
                            futuresDict.ix[sym].Group == group}\
                        for group in futuresDict.Group.unique()}
    return groupdict

def getComponents():
    ComponentsDict ={
                    'Off':['None'],
                    'Previous':['prevACT'],
                    'Anti-Previous':['AntiPrevACT'],
                    'RiskOn':['RiskOn'],
                    'RiskOff':['RiskOff'],
                    'Custom':['Custom'],
                    'Anti-Custom':['AntiCustom'],
                    '50/50':['0.75LastSIG'],
                    'LowestEquity':['0.5LastSIG'],
                    'HighestEquity':['1LastSIG'],
                    'AntiHighestEquity':['Anti1LastSIG'],
                    'Anti50/50':['Anti0.75LastSIG'],
                    'AntiLowestEquity':['Anti0.5LastSIG'],
                    'Seasonality':['LastSEA'],
                    'Anti-Seasonality':['AntiSEA'],
                    }
    return ComponentsDict

def updateMeta():
    readConn = getBackendDB()
    mcdate=MCdate()
    timetables = pd.read_sql('select * from timetable', con=readConn, index_col='Desc')
    futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='IBsym')
    if mcdate not in timetables.columns:
        print('Running MOC to get new mcdate...')
        start_moc()
        mcdate=timetables.drop(['Date','timestamp'],axis=1).columns[-1]
        triggers = pd.DataFrame(timetables[mcdate].ix[[x for x in timetables.index if 'trigger' in x]].copy())
        triggers[mcdate] = 'Not Available'
    else:
        triggers = pd.DataFrame(timetables[mcdate].ix[[x for x in timetables.index if 'trigger' in x]].copy())
    triggers.index=[x.split()[0] for x in triggers.index]
    triggers.columns = [['Triggertime']]
    triggers['Group']=futuresDict.ix[triggers.index].Group.values
    triggers['Date']=mcdate
    record = MetaData(components=json.dumps(getComponents()), triggers=json.dumps(triggers.transpose().to_dict()),\
                                    mcdate=mcdate,\
                                    timestamp=getTimeStamp())
    record.save()

def get_order_status():
    readConn = getBackendDB()
    orderstatus_dict={}
    accounts = ['v4micro', 'v4mini', 'v4futures']

    for account in accounts:
        col_order= ['broker','account','order_type','contract','selection','openedWhen','system_signal',\
                'broker_position','signal_check','system_qty','broker_qty','qty_check']
        df=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by contract' % ('checkSystems_'+account),\
                con=readConn)
        df['system_signal'] = np.where(df['system_signal'] == 1, 'LONG', 'SHORT')
        df['broker_position'] = np.where(df['broker_position'] == 1, 'LONG', 'SHORT')
        orderstatus_dict[account]=df[col_order].to_html()
    return orderstatus_dict

def getAccountValues():
    readConn = getBackendDB()
    mcdate = MCdate()
    eastern = timezone('US/Eastern')
    utc = timezone('UTC')
    accountvalues = {}
    urpnls = {}

    # ib
    ib_equity = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'NetLiquidation\'', \
                            con=readConn, index_col='Desc')
    timestamp = utc.localize(dt.utcfromtimestamp(ib_equity.timestamp)).astimezone(eastern)
    accountvalue = int(float(ib_equity.value[0]))
    accountvalues['v4futures'] = {
        'col1title': 'Account Value', 'col1value': accountvalue, \
        'col2title': 'Timestamp', 'col2value': timestamp.strftime('%Y-%m-%d %I:%M:%S %p EST')
    }

    ib_urpnl = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'UnrealizedPnL\' and currency=\'BASE\'', \
                           con=readConn, index_col='Desc')
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
    for system in c2_equity.drop(['v4futures'], axis=0).index:
        timestamp = c2_equity.ix[system].updatedLastTimeET
        accountvalue = int(c2_equity.ix[system].modelAccountValue)
        urpnl = int(c2_equity.ix[system].equity)

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

    record = AccountData(value1=json.dumps(urpnls), value2=json.dumps(accountvalues), mcdate=mcdate, \
                         timestamp=getTimeStamp())
    record.save()
          