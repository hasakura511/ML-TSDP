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


quantity=futuresDF_boards[currentdate][qtydict[account]].copy()
quantity.ix[[sym for sym in quantity.index if sym not in active_symbols[account]]]=0
futuresDF_boards[currentdate]['chgValue'] =  futuresDF_boards[currentdate].LastPctChg*\
                                                            futuresDF_boards[currentdate].contractValue*\
                                                            quantity
IB2CSI_multiplier_adj={
    'HG':100,
    'SI':100,
    'JPY':100,
    }

c2contractSpec = {
'AC':['@AC',fxDict['USD'],29000,'energy',1,1],
'AD':['@AD',fxDict['USD'],100000,'currency',1,1],
'AEX':['AEX',fxDict['EUR'],200,'index',1,1],
'BO':['@BO',fxDict['USD'],600,'grain',1,1],
'BP':['@BP',fxDict['USD'],62500,'currency',1,1],
'C':['@C',fxDict['USD'],50,'grain',1,1],
'CC':['@CC',fxDict['USD'],10,'soft',1,1],
'CD':['@CD',fxDict['USD'],100000,'currency',1,1],
'CGB':['CB',fxDict['CAD'],1000,'rates',-1,1],
'CL':['QCL',fxDict['USD'],1000,'energy',1,1],
'CT':['@CT',fxDict['USD'],500,'soft',1,1],
'CU':['@EU',fxDict['USD'],125000,'currency',1,1],
'DX':['@DX',fxDict['USD'],1000,'currency',-1,-1],
'EBL':['BD',fxDict['EUR'],1000,'rates',-1,1],
'EBM':['BL',fxDict['EUR'],1000,'rates',-1,1],
'EBS':['EZ',fxDict['EUR'],1000,'rates',-1,1],
'ED':['@ED',fxDict['USD'],2500,'rates',-1,1],
'EMD':['@EMD',fxDict['USD'],100,'index',1,1],
'ES':['@ES',fxDict['USD'],50,'index',1,1],
'FC':['@GF',fxDict['USD'],500,'meat',1,-1],
'FCH':['MT',fxDict['EUR'],10,'index',1,1],
'FDX':['DXM',fxDict['EUR'],5,'index',1,1],
'FEI':['IE',fxDict['EUR'],2500,'rates',-1,1],
'FFI':['LF',fxDict['GBP'],10,'index',1,1],
'FLG':['LG',fxDict['GBP'],1000,'rates',-1,1],
'FSS':['LL',fxDict['GBP'],1250,'rates',-1,-1],
'FV':['@FV',fxDict['USD'],1000,'rates',-1,1],
'GC':['QGC',fxDict['USD'],100,'metal',-1,1],
'HCM':['HHI',fxDict['HKD'],50,'index',1,1],
'HG':['QHG',fxDict['USD'],250,'metal',1,1],
'HIC':['HSI',fxDict['HKD'],50,'index',1,1],
'HO':['QHO',fxDict['USD'],42000,'energy',1,1],
'JY':['@JY',fxDict['USD'],125000,'currency',-1,1],
'KC':['@KC',fxDict['USD'],375,'soft',1,1],
'KW':['@KW',fxDict['USD'],50,'grain',1,1],
'LB':['@LB',fxDict['USD'],110,'soft',1,-1],
'LC':['@LE',fxDict['USD'],400,'meat',1,-1],
'LCO':['EB',fxDict['USD'],1000,'energy',1,1],
'LGO':['GAS',fxDict['USD'],100,'energy',1,1],
'LH':['@HE',fxDict['USD'],400,'meat',1,-1],
'LRC':['LRC',fxDict['USD'],10,'soft',1,1],
'LSU':['QW',fxDict['USD'],50,'soft',1,1],
'MEM':['@MME',fxDict['USD'],50,'index',1,1],
'MFX':['IB',fxDict['EUR'],10,'index',1,1],
'MP':['@PX',fxDict['USD'],500000,'currency',1,1],
'MW':['@MW',fxDict['USD'],50,'grain',1,1],
'NE':['@NE',fxDict['USD'],100000,'currency',1,1],
'NG':['QNG',fxDict['USD'],10000,'energy',1,1],
'NIY':['@NKD',fxDict['JPY'],500,'index',1,-1],
'NQ':['@NQ',fxDict['USD'],20,'index',1,1],
'O':['@O',fxDict['USD'],50,'grain',1,-1],
'OJ':['@OJ',fxDict['USD'],150,'soft',1,1],
'PA':['QPA',fxDict['USD'],100,'metal',1,1],
'PL':['QPL',fxDict['USD'],50,'metal',-1,1],
'RB':['QRB',fxDict['USD'],42000,'energy',1,1],
'RR':['@RR',fxDict['USD'],2000,'grain',1,-1],
'RS':['@RS',fxDict['CAD'],20,'grain',1,-1],
'S':['@S',fxDict['USD'],50,'grain',1,-1],
'SB':['@SB',fxDict['USD'],1120,'soft',1,1],
'SF':['@SF',fxDict['USD'],125000,'currency',1,1],
'SI':['QSI',fxDict['USD'],50,'metal',-1,1],
'SIN':['IN',fxDict['USD'],2,'index',1,1],
'SJB':['BB',fxDict['JPY'],100000,'rates',-1,1],
'SM':['@SM',fxDict['USD'],100,'grain',1,-1],
'SMI':['SW',fxDict['CHF'],10,'index',1,1],
'SSG':['SS',fxDict['SGD'],200,'index',1,-1],
'STW':['TW',fxDict['USD'],100,'index',1,1],
'SXE':['EX',fxDict['EUR'],10,'index',1,1],
'TF':['@TFS',fxDict['USD'],100,'index',1,1],
'TU':['@TU',fxDict['USD'],2000,'rates',-1,1],
'TY':['@TY',fxDict['USD'],1000,'rates',-1,1],
'US':['@US',fxDict['USD'],1000,'rates',-1,1],
'VX':['@VX',fxDict['USD'],1000,'index',-1,-1],
'W':['@W',fxDict['USD'],50,'grain',1,1],
'YA':['AP',fxDict['AUD'],25,'index',1,-1],
'YB':['HBS',fxDict['AUD'],2400,'rates',-1,1],
'YM':['@YM',fxDict['USD'],5,'index',1,1],
'YT2':['HTS',fxDict['AUD'],2800,'rates',-1,1],
'YT3':['HXS',fxDict['AUD'],8000,'rates',-1,1],
    }

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