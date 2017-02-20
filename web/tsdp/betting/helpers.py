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
from django import forms
from .models import MetaData, AccountData
from .start_moc import get_newtimetable, run_checksystems

class LogFiles(object):
    def __init__(self, filename):
        self.filename = filename

    def display_text_file(self):
        with open(self.filename) as fp:
            return fp.read()

def getBackendDB():
    dbPath = '/ML-TSDP/data/futures.sqlite3'
    readConn = sqlite3.connect(dbPath)
    return readConn

def get_logfiles(search_string='', exclude=False):
    search_dir = "/logs/"
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    if exclude:
        files = [os.path.join(search_dir, f) for f in files if search_string not in f]  # add path to each file
    else:
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
        get_newtimetable()
        mcdate=timetables.drop(['Date','timestamp'],axis=1).columns[-1]
        triggers = pd.DataFrame(timetables[mcdate].ix[[x for x in timetables.index if 'trigger' in x]].copy())
        triggers[mcdate] = 'Not Available'
    else:
        if timetables[mcdate].dropna().shape[0] == timetables.shape[0]:
            triggers = pd.DataFrame(timetables[mcdate].ix[[x for x in timetables.index if 'trigger' in x]].copy())
        else:
            triggers = pd.DataFrame(timetables[timetables.columns[0]].ix[[x for x in timetables.index if 'trigger' in x]].copy())

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
    futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='C2sym')

    orderstatus_dict={}
    slippage_files={}
    accounts = ['v4micro', 'v4mini', 'v4futures']

    def conv_sig(signals):
        sig = signals.copy()
        sig[sig == -1] = 'SHORT'
        sig[sig == 1] = 'LONG'
        sig[sig == 0] = 'NONE'
        return sig.values

    for account in accounts:
        col_order= ['broker','account','contract','description','openedWhen','urpnl',\
                'broker_position','broker_qty','signal_check','qty_check','selection','order_type']
        #webSelection=pd.read_sql('select * from webSelection where timestamp=\
        #        (select max(timestamp) from webSelection)'
        #bet = eval(webSelection.selection[0])[account][0]
        df=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by c2sym' % ('checkSystems_'+account),\
                con=readConn)
        df['system_signal'] = conv_sig(df['system_signal'])
        df['broker_position'] = conv_sig(df['broker_position'])
        desc_list=futuresDict.ix[[x[:-2] for x in df.contract.values]].Desc.values
        df['description']=[re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
        orderstatus_dict[account]=df[col_order].to_html()
        csidate = pd.read_sql('select distinct csiDate from slippage where Name=\'{}\''.format(account), con=readConn).csiDate.tolist()[-1]
        slippage_files[account]=str(csidate)
        if account == "v4futures":
            col_order=['desc','contracts','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency','bet','ordertype','status','Date']
            df = pd.read_sql('select * from (select * from %s\
                    order by timestamp ASC) group by ibsym' % ('checkSystems_ib_' + account), \
                             con=readConn)
            desc_list = [futuresDict.reset_index().set_index('IBsym').ix[x].Desc for x in df.ibsym]
            df['desc'] = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
            orderstatus_dict[account+'_ib'] = df[col_order].to_html()
            csidate = pd.read_sql('select distinct Date from ib_slippage where Name=\'{}\''.format(account),
                                  con=readConn).Date.tolist()[-1]
            slippage_files[account+'_ib'] = str(csidate)


    return slippage_files, orderstatus_dict

def getAccountValues():
    readConn = getBackendDB()
    mcdate = MCdate()
    eastern = timezone('US/Eastern')
    utc = timezone('UTC')
    now = dt.now(get_localzone())
    now = now.astimezone(eastern)
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

    #run checksystems and update values if last update >=1 day and not a T,W,TH,F,SA
    if (now-eastern.localize(c2_equity.updatedLastTimeET[0])).days>=1 and (now.weekday()>0 and now.weekday()<5):
        run_checksystems()

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

def get_detailed_timetable():
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
    futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='IBsym')
    timetables = pd.read_sql('select * from timetable', con=readConn, index_col='Desc').drop(['Date', 'timestamp'], axis=1)

    if mcdate in timetables and timetables[mcdate].dropna().shape[0]==timetables.shape[0]:
        ttdate = mcdate
        # filter out any dates that have passed
        ttdates = [x for x in timetables.columns if int(x) >= int(mcdate)]
        timetables = timetables[ttdates]
    else:
        # use old dates
        ttdate = timetables.columns[0]

    timetableDF = pd.DataFrame()
    for idx, [sym, value] in enumerate([x.split() for x in timetables.index]):
        idx2 = sym + ' ' + value
        timestamp = timetables.ix[idx2].ix[ttdate]
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

    col_order = ['group', 'immediate order type', 'open', 'close', 'trigger']
    # timetableDF=timetableDF[col_order]
    # print timetableDF
    # ttDict={account:timetableDF.ix[active_symbols_ib[account]].to_html() for account in active_symbols_ib}
    ttDict = {}
    for account in active_symbols_ib:
        df = timetableDF.ix[active_symbols_ib[account]]
        df['group'] = futuresDict.ix[df.index].Group
        desc_list = futuresDict.ix[df.index].Desc.values
        df.index = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
        ttDict[account] = df[col_order].sort_values(by='group').to_html()
    return ttDict

def get_overview():
    readConn = getBackendDB()
    accounts = ['v4micro', 'v4mini', 'v4futures']
    overviewDict={}
    for account in accounts:
        totalsDF = pd.read_sql('select * from {}'.format('totalsDF_board_' + account), con=readConn, index_col='Date')
        date = str(totalsDF.index[-1])
        overviewDict[account]=date
    return overviewDict

def archive_dates():
    readConn = getBackendDB()
    dates = pd.read_sql('select distinct Date from futuresATRhist', con=readConn).Date.tolist()
    startdate = 20170106
    datetup=[(dt.strptime(str(x), '%Y%m%d').strftime('%A, %b %d, %Y'),\
              dt.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d')) for x in\
             sorted(dates[dates.index(startdate):], reverse=True)]
    return datetup
    

def get_blends(cloc=None, list_boxstyles=None):
    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    if cloc == None:
        cloc = UserSelection.default_cloc

    if list_boxstyles == None:
        list_boxstyles = UserSelection.default_list_boxstyles
    else:
        list_boxstyles = [d for d in list_boxstyles if not is_int(d.keys()[0])]

    # print([cl.keys()[0] for cl in cloc])
    component_styles = {bs.keys()[0]: bs.values()[0] for bs in list_boxstyles if
                        bs.keys()[0] in [cl.keys()[0] for cl in cloc]}
    component_names = {cl.keys()[0]: cl.values()[0] for cl in cloc}

    # get from board js
    h_components = 6
    v_components = 6
    v_components_width = 2
    h_components_width = outside_components = 2
    table_height = 3

    table_width = num_v_component_boxes = h_components * outside_components
    num_h_component_boxes = outside_components * table_height
    total_boxes = table_height * table_width
    start_vert = outside_components + h_components
    end_vert = outside_components + h_components + v_components

    # figure this out later..
    # mod:boxkeys
    # vboxdict={0:['c9','c10'].
    #                2:['c11','c12'],
    #                1:['c13','c14'],
    #            }
    vboxdict = {}
    vboxlist = [x for x in range(table_height - 1, 0, -1)]
    vboxlist.insert(0, 0)
    for x in vboxlist:
        vboxdict[x] = []
        for y in range(0, v_components_width):
            start_vert += 1
            # print x,y,start_vert
            vboxdict[x].append('c' + str(start_vert))

    # 1-6 +c3 ... 31-36 +c8
    boxidDict = {}
    for boxid in range(1, total_boxes + 1):
        # print
        h_component = int(math.ceil(boxid / float(num_h_component_boxes)))
        boxidDict[str(boxid)] = ['c' + str(h_component + outside_components)]
        o_component = int(math.ceil(boxid / float(table_height))) - outside_components * (h_component - 1)
        boxidDict[str(boxid)] += ['c' + str(o_component)]
        boxidDict[str(boxid)] += vboxdict[boxid % table_height]

    boxstyleDict = {boxid: [component_styles[x] for x in boxidDict[boxid] if component_names[x] is not 'None'] for
                    boxid
                    in boxidDict}

    blendedboxstyleDict = {}
    for boxid, list_of_styles in boxstyleDict.items():
        if len(list_of_styles) > 0:
            # fillhex_test={}
            R = 0
            G = 0
            B = 0
            for i, style in enumerate(list_of_styles):
                blendedstyle = style.copy()
                # fillhex_test[i]=('%02x%02x%02x' % (int(style['fill-R']), int(style['fill-G']), int(style['fill-B']))).upper()
                R += int(blendedstyle['fill-R'])
                G += int(blendedstyle['fill-G'])
                B += int(blendedstyle['fill-B'])
                # print i, blendedstyle, R, G, B
            i += 1
            BR = int(R / float(i))
            BG = int(G / float(i))
            BB = int(B / float(i))
            fillhex = ('%02x%02x%02x' % (BR, BG, BB)).upper()
            # print i, BR, BG, BB, fillhex
            # blended = blend_colors(list_of_blendedstyles)
            L = 0
            ldict = {'r': 0.2126, 'g': 0.7152, 'b': 0.0722}
            for color, value in [('r', BR), ('g', BG), ('b', BB)]:
                c = value / 255.0
                if c <= 0.03928:
                    c = c / 12.92
                else:
                    c = ((c + 0.055) / 1.055) ** 2.4
                L += c * ldict[color]
            textcolor = '000000' if L > 0.179 else 'FFFFFF'
            # print L, textcolor

            blendedstyle.update({
                'fill-R': str(BR),
                'text-color': textcolor,
                'fill-Hex': fillhex,
                'fill-G': str(BG),
                'fill-B': str(BB),
                'text': str(boxid),
                # 'text-size': '24',
                # 'text-blendedstyle': 'bold',
                # 'text-font': 'Book Antigua',
                # 'fill-colorname': 'blended'
            })
            blendedboxstyleDict[boxid] = blendedstyle
            # print boxid, style
        else:
            blendedboxstyleDict[boxid] = {'filename': '',
                                          'fill-B': '255',
                                          'fill-G': '255',
                                          'fill-Hex': 'FFFFFF',
                                          'fill-R': '255',
                                          'text': '',
                                          'text-color': 'FFFFFF',
                                          'text-font': 'Arial Black',
                                          'text-size': '18',
                                          'text-style': 'bold'}
    keys = blendedboxstyleDict.keys()
    keys.sort(key=int)
    list_boxstyles += [{key: blendedboxstyleDict[key]} for key in keys]

    filename = 'boxstyles_data.json'
    with open(filename, 'w') as f:
        json.dump(list_boxstyles, f)
    print 'Saved', filename
    # return cloc, list_boxstyles