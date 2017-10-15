import os
from os import listdir
from os.path import isfile, join
import random
import sys
from subprocess import Popen, PIPE, check_output
import pandas as pd
import numpy as np
import threading
import time
import logging
import copy
import calendar
#import get_feed2 as feed
from pytz import timezone
from dateutil.parser import parse
import datetime
import traceback
from ibapi.wrapper_v5 import IBWrapper, IBclient

import slackweb
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
slackhook='https://hooks.slack.com/services/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'
slack = slackweb.Slack(url=slackhook)
slack_channel="#logs"

start_time = time.time()
callback = IBWrapper()
client=IBclient(callback, port=7496, clientid=0)
durationStr ='2 D'
barSizeSetting='30 mins'
whatToShow='TRADES'
interval='1d'
minDataPoints = 5
eastern=timezone('US/Eastern')
endDateTime=dt.now(get_localzone())
endDateTime=endDateTime.astimezone(eastern)
endDateTime=endDateTime.strftime("%Y%m%d %H:%M:%S EST")    
data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')

print 'IB get history seetings:', durationStr, barSizeSetting, whatToShow
#feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
feeddata=pd.read_csv(feedfile,index_col='ibsym')
#update margin function here
feeddata['Date']=csidate

dbPath=dbPath2='./data/futures.sqlite3'
runPath='./run_futures_live.py'
runPath2=['python','./vol_adjsize_live.py','1']
runPath3=['python','./proc_signal_v4_live.py','1']
runPath4=['python','./check_systems_live.py','1']
logPath='/logs/'
dataPath='./data/'
portfolioPath = './data/portfolio/'
savePath='./data/'
pngPath = './web/tsdp/betting/static/images/'
savePath2 = './data/portfolio/'
systemPath =  './data/systems/'
feedfile='./data/systems/system_ibfeed.csv'
#systemfile='./data/systems/system_v4micro.csv'
timetablePath=   './data/systems/timetables/'
#feedfile='D:/ML-TSDP/data/systems/system_ibfeed_fx.csv'
csiDataPath=  './data/csidata/v4futures2/'
csiDataPath2=  './data/csidata/v4futures3/'
csiDataPath3=  './data/csidata/v4futures4/'
signalPathDaily =  './data/signals/'
signalPathMOC =  './data/signals2/'
logging.basicConfig(filename='/logs/broker_live_'+dt.now().strftime('%Y%m%d-%H%M%S')+'.log',level=logging.DEBUG)


#def create_execDict(feeddata, systemdata):

print 'loading contract details from file'
contractsDF=pd.read_csv(systemPath+'ib_contracts.csv', index_col='ibsym')
    
feeddata=feeddata.reset_index()
for i in feeddata.index:
    
    #print 'Read: ',i
    system=feeddata.ix[i]
    c2sym=system.c2sym
    ibsym=system.ibsym   
    index = systemdata[systemdata.c2sym2==c2sym].index[0]
    #find the current contract

    #print system
    contract = Contract()
    
    if system['ibtype'] == 'CASH':
        #fx
        symbol=system['ibsym']+system['ibcur']
        contract.symbol=system['ibsym']
        ccontract = ''
    else:
        #futures
        contract.expiry = getContractDate(system.c2sym, systemdata)
        symbol=contract.symbol= system['ibsym']
        contract.multiplier = str(system['multiplier'])

    #contract.symbol = system['ibsym']
    contract.secType = system['ibtype']
    contract.exchange = system['ibexch']
    contract.currency = system['ibcur']
    '''
    print i+1, contract.symbol, contract.expiry
    if downloadtt:
        contractInfo=client.get_contract_details(contract)
        #print contractInfo
        contractsDF=contractsDF.append(contractInfo)
        execDict[symbol+contractInfo.expiry[0]]=['PASS', 0, contract]
        systemdata.set_value(index, 'ibcontract', symbol+contractInfo.expiry[0])
    else:
	'''
    execDict[contractsDF.ix[symbol].contracts]=['PASS', 0, contract]
    systemdata.set_value(index, 'ibcontract', contractsDF.ix[symbol].contracts)
        
    #update system file with correct ibsym and contract expiry
    #print index, ibsym, contract.expiry, systemdata.columns
    systemdata.set_value(index, 'ibsym', ibsym)
    systemdata.set_value(index, 'ibcontractmonth', contract.expiry)
    

    #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, contract.expiry

#systemdata.to_csv(systemfile, index=False)
systemdata.to_sql(name='v4futures_moc_live', if_exists='replace', con=writeConn, index=False)
print '\nsaved v4futures_moc_live to', dbPath

if downloadtt:
    feeddata=feeddata.set_index('ibsym')
    contractsDF=contractsDF.set_index('symbol')
    contractsDF.index.name = 'ibsym'
    contractsDF['contracts']=[x+contractsDF.ix[x].expiry for x in contractsDF.index]
    contractsDF['Date']=csidate
    contractsDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    #print contractsDF.index
    #print feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur'],axis=1).head()
    contractsDF = pd.concat([ feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur','Date','timestamp'],axis=1),contractsDF], axis=1)
    try:
        contractsDF.to_sql(name='ib_contracts', con=writeConn, index=True, if_exists='replace', index_label='ibsym')
        print '\nsaved ib_contracts to',dbPath
    except Exception as e:
        #print e
        traceback.print_exc()
    if not debug:
        contractsDF.to_csv(systemPath+'ib_contracts.csv', index=True)
        print 'saved', systemPath+'ib_contracts.csv'
        
print '\nCreated exec dict with', len(execDict.keys()), 'contracts:'
print execDict.keys()
return execDict,contractsDF,

def refresh_all_histories(execDict):
    global client
    global feeddata
    global endDateTime
    for sym in execDict:
        data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
        tickerId=random.randint(100,9999)
        contract = execDict[sym][2]
        print sym, 'getting data from IB'
        data = client.get_history(endDateTime, contract, whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
        data.to_csv(csiDataPath2+feeddata.ix[sym].CSIsym2+'.csv', index=True)

execDict={}
contractsDF=pd.DataFrame()
tries = 0
while (len(execDict)  == 0 or len(contractsDF) == 0) and tries<5:
    try:
        execDict, contractsDF, futuresDF=create_execDict(feeddata, futuresDF)
    except Exception as e:
        #print e
        slack.notify(text='create_execDict: '+str(e), channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
        traceback.print_exc()
        tries+=1
        if tries==5:
            sys.exit('failed 5 times to get contract info')