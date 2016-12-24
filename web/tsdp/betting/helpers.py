from django import forms
import time
import datetime
from datetime import datetime as dt
from pytz import timezone
from tzlocal import get_localzone
import sqlite3
import pandas as pd
from .models import MetaData, AccountData
import calendar

def getBackendDB():
    dbPath = '/ML-TSDP/data/futures.sqlite3'
    readConn = sqlite3.connect(dbPath)
    return readConn

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
    triggers = pd.DataFrame(timetables[mcdate].ix[[x for x in timetables.index if 'trigger' in x]].copy())
    triggers.index=[x.split()[0] for x in triggers.index]
    triggers.columns = [['Triggertime']]
    triggers['Group']=futuresDict.ix[triggers.index].Group.values
    triggers['Date']=mcdate
    record = MetaData(components=getComponents(), triggers=triggers.transpose().to_dict(),mcdate=mcdate,\
                                    timestamp=getTimeStamp())
    record.save()
    
def getAccountValues():
    readConn = getBackendDB()
    mcdate = MCdate()
    eastern = timezone('US/Eastern')
    utc = timezone('UTC')
    accountvalues={}
    urpnls = {}
    
    #ib
    ib_equity = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'NetLiquidation\'',\
                            con=readConn, index_col='Desc')
    timestamp = utc.localize(dt.utcfromtimestamp(ib_equity.timestamp)).astimezone(eastern)
    accountvalue=int(float(ib_equity.value[0]))
    accountvalues['v4futures']={'timestamp':timestamp.strftime('%Y-%m-%d %I:%M:%S %p'), 'Account Value':accountvalue}
    
    ib_urpnl = pd.read_sql('select * from ib_accountData where timestamp=\
                (select max(timestamp) from ib_accountData) and Desc=\'UnrealizedPnL\' and currency=\'BASE\'',\
                           con=readConn, index_col='Desc')
    timestamp = utc.localize(dt.utcfromtimestamp(ib_urpnl.timestamp)).astimezone(eastern)
    urpnl=int(float(ib_urpnl.value[0]))
    urpnls['v4futures']={'timestamp':timestamp.strftime('%Y-%m-%d %I:%M:%S %p'), 'UnrealizedPnL':urpnl}
    
    #C2
    c2_equity = pd.read_sql('select * from (select * from c2_equity order by timestamp ASC) group by system',\
                                        con=readConn, index_col='system')
    c2_equity.updatedLastTimeET = pd.to_datetime(c2_equity.updatedLastTimeET)
    for system in c2_equity.drop(['v4futures'], axis=0).index:
        timestamp = c2_equity.ix[system].updatedLastTimeET.strftime('%Y-%m-%d %I:%M:%S %p')
        accountvalue=c2_equity.ix[system].modelAccountValue
        urpnl=c2_equity.ix[system].equity
        accountvalues[system]={'timestamp':timestamp, 'Account Value':accountvalue}
        urpnls[system]={'timestamp':timestamp, 'UnrealizedPnL':urpnl}
    '''
    c2_v4micro = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4micro\')', con=readConn)

    c2_v4mini = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4mini\')', con=readConn)

    c2_v4futures = pd.read_sql('select * from c2_portfolio where timestamp=\
          (select max(timestamp) from c2_portfolio where system=\'v4futures\')', con=readConn)
    '''
    
    record = AccountData(value1=urpnls, value2=accountvalues,mcdate=mcdate,\
                                    timestamp=getTimeStamp())
    record.save()
  
          