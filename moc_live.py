#import ibapi.futures_bars_1d as bars
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
#import get_feed2 as feed
from pytz import timezone
from dateutil.parser import parse
import datetime
import traceback
from ibapi.wrapper_v5 import IBWrapper, IBclient
from ibapi.place_order2 import place_orders as place_iborders
from swigibpy import Contract 
import pandas as pd
from time import gmtime, strftime, localtime, sleep
import json
import datetime
from pandas.io.json import json_normalize
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import logging
from swigibpy import EPosixClientSocket, ExecutionFilter, CommissionReport, Execution, Contract
from dateutil.parser import parse
import sqlite3
#currencyPairsDict=dict()
#prepData=dict()
start_time = time.time()
callback = IBWrapper()
client=IBclient(callback)

#systems = ['v4micro','v4mini','v4macro']

durationStr ='2 D'
barSizeSetting='30 mins'
whatToShow='TRADES'

filename=None
eastern=timezone('US/Eastern')
endDateTime=dt.now(get_localzone())
endDateTime=endDateTime.astimezone(eastern)
endDateTime=endDateTime.strftime("%Y%m%d %H:%M:%S EST")    
data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
tickerId=random.randint(100,9999)

interval='1d'
minDataPoints = 5

#python moc_live.py 1 1 1
#python moc_live.py live submitc2 submitIB
if len(sys.argv)==1:
    #systems = ['v4mini']
    systems = ['v4micro','v4mini','v4futures']
    debug=True
    showPlots=True
    submitIB=False
    submitC2=False
    triggertime = 30 #mins
    dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
    runPath='D:/ML-TSDP/run_futures_live.py'
    runPath2= ['python','D:/ML-TSDP/vol_adjsize_live.py']
    runPath3=  ['python','D:/ML-TSDP/proc_signal_v4_live.py','0']
    runPath4=['python','D:/ML-TSDP/check_systems_live.py','0']
    logPath='C:/logs/'
    dataPath='D:/ML-TSDP/data/'
    #portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    #savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath=portfolioPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    systemPath =  'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/'
    feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
    #systemfile='D:/ML-TSDP/data/systems/system_v4micro.csv'
    timetablePath=   'D:/ML-TSDP/data/systems/timetables_debug/'
    #feedfile='D:/ML-TSDP/data/systems/system_ibfeed_fx.csv'
    csiDataPath=  'D:/ML-TSDP/data/csidata/v4futures2/'
    csiDataPath2=  'D:/ML-TSDP/data/csidata/v4futures3/'
    csiDataPath3=  'D:/ML-TSDP/data/csidata/v4futures4_debug/'
    signalPathDaily =  'D:/ML-TSDP/data/signals/'
    signalPathMOC =  'D:/ML-TSDP/data/signals2/'
    logging.basicConfig(filename='C:/logs/ib_live.log',level=logging.DEBUG)
else:
    systems = ['v4micro','v4mini','v4futures']
    debug=False
    showPlots=False
    
    if sys.argv[2]=='1':
        submitC2=True
    else:
        submitC2=False
    
    if sys.argv[3]=='1':
        submitIB=True
    else:
        submitIB=False
    
    
    triggertime = 30 #mins
    dbPath='./data/futures.sqlite3'
    runPath='./run_futures_live.py'
    runPath2=['python','./vol_adjsize_live.py','1']
    runPath3=['python','./proc_signal_v4_live.py','1']
    runPath4=['python','./check_systems_live.py','1']
    logPath='/logs/'
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    savePath='./data/'
    pngPath = './data/results/'
    savePath2 = './data/portfolio/'
    systemPath =  './data/systems/'
    feedfile='./data/systems/system_ibfeed.csv'
    systemfile='./data/systems/system_v4micro.csv'
    timetablePath=   './data/systems/timetables/'
    #feedfile='D:/ML-TSDP/data/systems/system_ibfeed_fx.csv'
    csiDataPath=  './data/csidata/v4futures2/'
    csiDataPath2=  './data/csidata/v4futures3/'
    csiDataPath3=  './data/csidata/v4futures4/'
    signalPathDaily =  './data/signals/'
    signalPathMOC =  './data/signals2/'
    logging.basicConfig(filename='/logs/ib_live.log',level=logging.DEBUG)

tzDict = {
    'CST':'CST6CDT',
    'EST':'EST5EDT',
    }
days = {
                0:'Mon',
                1:'Tues',
                2:'Wed',
                3:'Thurs',
                4:'Fri',
                5:'Sat',
                6:'Sun',
                }
months = {
                'F':1,
                'G':2,
                'H':3,
                'J':4,
                'K':5,
                'M':6,
                'N':7,
                'Q':8,
                'U':9,
                'V':10,
                'X':11,
                'Z':12
                }
conn = sqlite3.connect(dbPath)

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        

def runInThread(sym, popenArgs):
    with open(logPath+sym+'.txt', 'w') as f:
        with open(logPath+sym+'_error.txt', 'w') as e:
            proc = Popen(popenArgs, stdout=f, stderr=e)
            proc.wait()
            #check_output(popenArgs)
            proc2= Popen(popenArgs2, stdout=f, stderr=e)
            proc2.wait()
            proc_orders(sym)
        return
    thread = threading.Thread(target=runInThread)
    #, args=(onExit, popenArgs))
    thread.start()
    # returns immediately after the thread starts
    return thread

def runThreads(threadlist):

    def runInThread(sym, popenArgs):
        print 'starting thread for', sym
        
        with open(logPath+sym+'.txt', 'w') as f:
            with open(logPath+sym+'_error.txt', 'w') as e:
                proc = Popen(popenArgs, stdout=f, stderr=e)
                proc.wait()
                print sym,'Done!'
                #check_output(popenArgs)
                #proc2= Popen(popenArgs2, stdout=f, stderr=e)
                #proc2.wait()
                #proc_orders(sym)
            return
            
    threads=[]
    for arg in threadlist:
        #print arg
        t = threading.Thread(target=runInThread, args=arg)
        threads.append(t)
        
     # Start all threads
    for x in threads:
        x.start()

     # Wait for all of them to finish
    for x in threads:
        x.join()

    
def get_ibfutpositions(portfolioPath):
    global client
    (account_value, portfolio_data)=client.get_IB_account_data()
    
    if len(account_value) != 0:
        accountSet=pd.DataFrame(account_value,columns=['desc','value','currency','account_id'])
        accountSet=accountSet.set_index(['desc'])
        filename=portfolioPath+'ib_account_value.csv'
        accountSet.to_csv(filename, index=True)
        print 'saved', filename
        if 'NetLiquidation' in accountSet.index:
            accountValue = accountSet.ix['NetLiquidation'].value        
            print 'Account value:', accountValue
    else:
        print 'Account value returned nothing'
        
    if len(portfolio_data) !=0:
        data=pd.DataFrame(portfolio_data,columns=['sym','exp','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
        dataSet=data[data.exp != '']
        dataSet=dataSet.set_index(['sym'])
        filename=portfolioPath+'ib_portfolio.csv'
        dataSet.to_csv(filename)
        print 'saved', filename
        print dataSet.shape[0],'futures positions found'
        return dataSet
    else:
        return 0

def create_execDict(feeddata, systemfile):
    global client
    execDict=dict()
    #need systemdata for the contract expiry
    systemdata=pd.read_csv(systemfile)
    systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    systemdata['CSIsym']=[x.split('_')[1] for x in systemdata.System]
    #openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    feeddata=feeddata.reset_index()
    for i in feeddata.index:
        
        #print 'Read: ',i
        system=feeddata.ix[i]
        #find the current contract

        #print system
        contract = Contract()
        
        if system['ibtype'] == 'CASH':
            #fx
            symbol=system['ibsym']+system['ibcur']
            contract.symbol=system['ibsym']
        else:
            #futures
            currentcontract = [x for i,x in enumerate(systemdata.c2sym) if x[:-2] == system.c2sym]
            if len(currentcontract)==1:
                #Z6
                ccontract = currentcontract[0][-2:]
                ccontract = 201000+int(ccontract[-1])*100+months[ccontract[0]]
                contract.expiry=str(ccontract)
            else:
                ccontract = ''
            symbol=contract.symbol= system['ibsym']
            contract.multiplier = str(system['multiplier'])

        #contract.symbol = system['ibsym']
        contract.secType = system['ibtype']
        contract.exchange = system['ibexch']
        contract.currency = system['ibcur']
        
        #update system file with correct ibsym and contract expiry
        c2sym=system.c2sym
        ibsym=system.ibsym   
        index = systemdata[systemdata.c2sym2==c2sym].index[0]
        #print index, ibsym, ccontract, systemdata.columns
        systemdata.set_value(index, 'ibsym', ibsym)
        systemdata.set_value(index, 'ibexpiry', ccontract)
        execDict[symbol]=['PASS', 0, contract]

        #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, ccontract
    systemdata.to_csv(systemfile, index=False)
    print 'updated', systemfile
      
    print 'Created exec dict with', len(execDict.keys()), 'symbols:'
    print execDict.keys()
    return execDict
   
def update_orders(feeddata, systemfile, execDict):
    global client
    global portfolioPath
    systemdata=pd.read_csv(systemfile)
    systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['CSIsym']=[x.split('_')[1] for x in systemdata.System]
    #openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    feeddata=feeddata.reset_index()
    portfolio=get_ibfutpositions(portfolioPath)
    if isinstance(portfolio, type(pd.DataFrame())):
        openPositions = portfolio.reset_index().groupby(['sym'])[['qty']].sum()
        for i in feeddata.index:
            system=feeddata.ix[i]
            c2sym=system.c2sym
            ibsym=system.ibsym   
            index = systemdata[systemdata.c2sym2==c2sym].index[0]

            if ibsym in openPositions.index:
                ib_pos_qty=openPositions.ix[ibsym].qty
            else:
                ib_pos_qty=0
            #ib_pos_qty=0
            #print ib_pos_qty
            ibquant = systemdata.ix[index].c2qty
            system_ibpos_qty=systemdata.ix[index].signal * ibquant
            #print 'ibq', type(ibquant), 'sysibq', type(system_ibpos_qty)
            #print( "system_ib_pos: " + str(system_ibpos_qty) ),
            #print( "ib_pos: " + str(ib_pos_qty) ),
            
            action='PASS'
            if system_ibpos_qty > ib_pos_qty:
                action = 'BOT'
                ibquant=int(system_ibpos_qty - ib_pos_qty)
                #print( 'BUY: ' + str(ibquant) )
                #place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
            if system_ibpos_qty < ib_pos_qty:
                action='SLD'
                ibquant=int(ib_pos_qty - system_ibpos_qty)
                #print( 'SELL: ' + str(ibquant) )
                #place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);         
            #print( action+': ' + str(ibquant) )
            execDict[ibsym][0]=action
            execDict[ibsym][1]=ibquant

            #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, ccontract
        #systemdata.to_csv(systemfile, index=False)
        #print 'saved', systemfile
          
        #print len(execDict.keys()), execDict.keys()
        return execDict
    else:
        print 'Could not get open positions from IB'
        return execDict

def get_orders(feeddata, systemfile):
    global client
    systemdata=pd.read_csv(systemfile)
    execDict=dict()

    systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['CSIsym']=[x.split('_')[1] for x in systemdata.System]
    openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    feeddata=feeddata.reset_index()
    for i in feeddata.index:
        
        #print 'Read: ',i
        system=feeddata.ix[i]
        #find the current contract

        #print system
        contract = Contract()
        
        if system['ibtype'] == 'CASH':
            #fx
            symbol=system['ibsym']+system['ibcur']
            contract.symbol=system['ibsym']
        else:
            #futures
            currentcontract = [x for i,x in enumerate(systemdata.c2sym) if x[:-2] == system.c2sym]
            if len(currentcontract)==1:
                #Z6
                ccontract = currentcontract[0][-2:]
                ccontract = 201000+int(ccontract[-1])*100+months[ccontract[0]]
                contract.expiry=str(ccontract)
            else:
                ccontract = ''
            symbol=contract.symbol= system['ibsym']
            contract.multiplier = str(system['multiplier'])

        #contract.symbol = system['ibsym']
        contract.secType = system['ibtype']
        contract.exchange = system['ibexch']
        contract.currency = system['ibcur']
        
        #update system file with correct ibsym and contract expiry
        c2sym=system.c2sym
        ibsym=system.ibsym   
        index = systemdata[systemdata.c2sym2==c2sym].index[0]
        #print index, ibsym, ccontract, systemdata.columns
        systemdata.set_value(index, 'ibsym', ibsym)
        systemdata.set_value(index, 'ibexpiry', ccontract)
        
        if ibsym in openPositions.index:
            ib_pos_qty=openPositions.ix[ibsym].qty
        else:
            ib_pos_qty=0
        #ib_pos_qty=0
        #print ib_pos_qty
        ibquant = systemdata.ix[index].c2qty
        system_ibpos_qty=systemdata.ix[index].signal * ibquant
        #print 'ibq', type(ibquant), 'sysibq', type(system_ibpos_qty)
        #print( "system_ib_pos: " + str(system_ibpos_qty) ),
        #print( "ib_pos: " + str(ib_pos_qty) ),
        
        action='PASS'
        if system_ibpos_qty > ib_pos_qty:
            action = 'BUY'
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            #print( 'BUY: ' + str(ibquant) )
            #place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
        if system_ibpos_qty < ib_pos_qty:
            action='SELL'
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            #print( 'SELL: ' + str(ibquant) )
            #place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);         
        #print( action+': ' + str(ibquant) )
        execDict[symbol]=[action, ibquant, contract]

        #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, ccontract
    #systemdata.to_csv(systemfile, index=False)
    #print 'saved', systemfile
      
    print len(execDict.keys()), execDict.keys()
    return execDict

def get_contractdf(execDict, systemPath):
    global client
    contracts = [x[2] for x in execDict.values()]
    contractDF=pd.DataFrame()
    for i,contract in enumerate(contracts):
        print i, contract.symbol,
        contractDF=contractDF.append(json_normalize(client.get_contract_details(contract)))
    contractDF.to_csv(systemPath+'ib_contracts.csv', index=False)
    return contractDF


def refresh_all_histories(execDict):
    global client
    global feeddata
    for sym in execDict:
        data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
        tickerId=random.randint(100,9999)
        contract = execDict[sym][2]
        print sym, 'getting data from IB'
        data = client.get_history(endDateTime, contract, whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
        data.to_csv(csiDataPath2+feeddata.ix[sym].CSIsym2+'.csv', index=True)
        
def refresh_history(sym, execDict):
    global client
    global feeddata
    data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
    tickerId=random.randint(100,9999)
    contract = execDict[sym][2]
    print 'getting data from IB...',
    data = client.get_history(endDateTime, contract, whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
    data.to_csv(csiDataPath2+feeddata.ix[sym].CSIsym2+'.csv', index=True)
    return data
    
def get_tradingHours(sym, contractsDF):
    global triggertime
    fmt = '%Y-%m-%d %H:%M'
    dates = contractsDF.ix[sym].tradingHours.split(";")
    tz = timezone(tzDict[contractsDF.ix[sym].timeZoneId[:3]])
    
    thDict = {}
    for th in dates:
        thlist = th.split(':')
        date = thlist[0]
        if thlist[1] =='CLOSED':
            continue
        else:
            #print thlist
            openclosetimes = thlist[1].split('-')
            opentime = openclosetimes[0]
            opendate=dt(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])-1,\
                hour=int(opentime[0:2]), minute=int(opentime[2:4]), tzinfo=tz).astimezone(timezone(tzDict['EST']))
            closetime = openclosetimes[-1]
            closedate=dt(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]),\
                hour=int(closetime[0:2]), minute=int(closetime[2:4]), tzinfo=tz).astimezone(timezone(tzDict['EST']))
            triggerdate=closedate-datetime.timedelta(minutes=triggertime)
            thDict[date]=[opendate.strftime(fmt),closedate.strftime(fmt), triggerdate.strftime(fmt)]
    return thDict
        
def lastCsiDownloadDate():
    global csiDataPath
    datafiles = os.listdir(csiDataPath)
    dates = []
    for f in datafiles:
        lastdate = pd.read_csv(csiDataPath+f, index_col=0).index[-1]
        if lastdate not in dates:
            dates.append(lastdate)
            
    return max(dates)
    
def filterIBexec():
    global client
    global feeddata
    global csiDataPath3
    executions=pd.DataFrame(client.get_executions())
    executions=executions.set_index('symbol')
    executions['CSIsym2']=[feeddata.ix[sym].CSIsym2 for sym in executions.index]
    index = executions.reset_index().groupby(['CSIsym2'])['times'].transform(max)==executions.times
    executions= executions.reset_index().ix[index].set_index('CSIsym2')
    datafiles = os.listdir(csiDataPath3)

    for f in [f for f in datafiles if f.split('_')[0] in executions.index]:
        sym = f.split('_')[0]
        lastdate = pd.read_csv(csiDataPath3+f, index_col=0).index[-1]
        executions.set_value(sym, 'lastAppend', dt.strptime(str(lastdate),'%Y%m%d'))

    executions.times = pd.to_datetime(executions.times)
    executions = executions[executions.times >= executions.lastAppend].reset_index().set_index('symbol')
    #combine orders
    
    executions.to_csv(portfolioPath+'ib_exec_last.csv', index=True)
    print 'saved', portfolioPath+'ib_exec_last.csv'
    return executions
    
def get_timetable(execDict, systemPath):
    global client
    global csiDataPath
    #to be run after csi download
    
    contractsDF = get_contractdf(execDict, systemPath)
    contractsDF=contractsDF.set_index('symbol')
    for i,sym in enumerate(contractsDF.index):
        thDict = get_tradingHours(sym, contractsDF)
        if i == 0:
            timetable=pd.DataFrame(thDict,index=[sym+' open',sym+' close',sym+' trigger'])
        else:
            timetable=pd.concat([timetable, pd.DataFrame(thDict,index=[sym+' open',sym+' close',sym+' trigger'])], axis=0)
    csidate=lastCsiDownloadDate()
    filedate=[d for d in timetable.columns.astype(int) if d>csidate][0]
    filename=timetablePath+str(filedate)+'.csv'
    timetable.to_csv(filename, index=True)
    print 'saved', filename
    return timetable
        
def find_triggers(feeddata, execDict):

    eastern=timezone(tzDict['EST'])
    endDateTime=dt.now(get_localzone())
    #endDateTime=dt.now(get_localzone())+datetime.timedelta(days=5)
    endDateTime=endDateTime.astimezone(eastern)

    #load timetable
    ttfiles = os.listdir(timetablePath)
    ttdates = []
    for f in ttfiles:
        if '.csv' in f and is_int(f.split('.')[0]):
            ttdates.append(int(f.split('.')[0]))
        
    csidate=lastCsiDownloadDate()
    ttdate=max(ttdates)
    if ttdate>csidate:
        #timetable file date is greater than the csi download date. 
        loaddate=str(ttdate)
    else:
        #get a new timetable
        print 'csidate',csidate, '>=', 'ttdate', ttdate, 'getting new timetable'
        timetable = get_timetable(execDict, systemPath)
        loaddate=str([d for d in timetable.columns.astype(int) if d>csidate][0])
        
    filename=timetablePath+loaddate+'.csv'
    timetable=pd.read_csv(filename, index_col=0)
    triggers=timetable.ix[[i for i in timetable.index if 'trigger' in i]][loaddate]
    threadlist=[]
    signalFilesMOC = [ f for f in listdir(signalPathMOC) if isfile(join(signalPathMOC,f)) ]
    signalFilesDaily =[ f for f in listdir(signalPathDaily) if isfile(join(signalPathDaily,f)) ]
    for t in triggers.index:
        ibsym=t.split()[0]
        csiFileSym=feeddata.ix[ibsym].CSIsym2
        csiRunSym=feeddata.ix[ibsym].CSIsym
        fmt = '%Y-%m-%d %H:%M'    
        tdate=dt.strptime(triggers.ix[t],fmt).replace(tzinfo=eastern)
        if endDateTime>tdate:
            #print 'checking trigger:',
            filename = csiDataPath3+csiFileSym+'_B.CSV'
            if not os.path.isfile(filename) or os.path.getsize(filename)==0:
                #create new file
                print csiRunSym, 'file not found appending data'
                dataNotAppended = True
            else:
                #check csiDataPath3 for last date
                data = pd.read_csv(filename, index_col=0, header=None)
                lastdate=data.index[-1]
                
                symSignalFilesDaily=[x for x in signalFilesDaily if '_'+csiRunSym+'_' in x]
                symSignalFilesMOC=[x for x in signalFilesMOC if '_'+csiRunSym+'_' in x]
                
                for f in [x for x in symSignalFilesDaily if x not in symSignalFilesMOC]:
                    #is signal file dosen't exist copy a portion of the old one.
                    pd.read_csv(signalPathDaily+f).iloc[-2:].to_csv(signalPathMOC+f, index=False)
                
                #if int(loaddate) > lastdate and int(loaddate) > int(lastsignaldate):
                if int(loaddate) > lastdate:
                    print csiRunSym,'appending.. data has not yet been appended',
                    print 'loaddate', loaddate, '>', 'lastdate',
                    #print 'loaddate', loaddate, '>', 'lastdate',lastdate,'lastsignaldate', lastsignaldate
                    dataNotAppended=True
                else:
                    #if int(loaddate) <= lastdate:
                    print csiRunSym,'skipping append.. data has already been appended',
                    #if int(loaddate) <= int(lastsignaldate):
                    #    print csiRunSym,'skipping append.. signal has been generated',
                    print 'loaddate', loaddate, '<', 'lastdate',lastdate
                    dataNotAppended=False
            #append data if M-F, not a holiday and if the data hasn't been appended yet. US MARKETS EST.
            dayofweek = endDateTime.date().weekday()
            
            if dayofweek<5 and dataNotAppended:
            #if dataNotAppended:
                #append new bar
                runsystem = append_data(ibsym, timetable, loaddate)
                if runsystem:
                    print 'data appended running system',
                    if debug==True:
                        print 'debug mode'
                        popenArgs = ['python', runPath,csiRunSym]
                        #popenArgs2 = ['python', runPath2, csiFileSym,'0']
                        threadlist.append((csiRunSym,popenArgs))
                    else:
                        print 'live mode'
                        popenArgs = ['python', runPath,csiRunSym,'0']
                        #popenArgs2 = ['python', runPath2, csiFileSym,'1']
                        threadlist.append((csiRunSym,popenArgs))
                else:
                    print 'skipping runsystem append_data returned 0'
            else:
                if dayofweek>=5:
                    print 'skipping append.. day of week', days[dayofweek]

                
        else:
            print csiRunSym,'not triggered: next trigger',tdate,'now', endDateTime
    return threadlist
                
def append_data(sym, timetable, loaddate):
    global feeddata
    global execDict
    global client
    global csiDataPath
    global csiDataPath3
    #datafiles = os.listdir(csiDataPath)
    fmt = '%Y-%m-%d %H:%M'
    #create new bar
    data = refresh_history(sym, execDict)
    data.index = data.index.to_datetime()
    opentime = dt.strptime(timetable[loaddate][sym+' open'],fmt)
    closetime = dt.strptime(timetable[loaddate][sym+' close'],fmt)
    mask = (data.index >= opentime) & (data.index <= closetime)
    data2=data.ix[mask]
    if data2.shape[0]>0:
        newbar = pd.DataFrame({}, columns=['Date', 'Open','High','Low','Close','Volume','OI','R','S']).set_index('Date')
        newbar.loc[loaddate] = [data.Open[0],max(data2.High), min(data2.Low),data2.Close[-1],data2.Volume.sum(),np.nan,np.nan,np.nan]
        #load old bar
        csisym=feeddata.ix[sym].CSIsym2
        filename = csiDataPath+csisym+'_B.CSV'
        if os.path.isfile(filename):
            csidata=pd.read_csv(filename, index_col=0, header=None)
            csidata.index.name = 'Date'
            csidata.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            filename = csiDataPath3+csisym+'_B.CSV'
            csidata.append(newbar).fillna(method='ffill').to_csv(filename, header=False, index=True)
            print 'saved', filename
            return True
        else:
            print filename, 'not found. terminating.',
            return False
    else:
        print 'no data found between', opentime, closetime,
        return False
        


if __name__ == "__main__":
    print durationStr, barSizeSetting, whatToShow
    feeddata=pd.read_csv(feedfile,index_col='ibsym')
    systemfile=systemPath+'system_v4futures_live.csv'
    #systemfile=systemPath+'system_'+sys+'_live.csv'
    execDict=create_execDict(feeddata, systemfile)
    threadlist=find_triggers(feeddata, execDict)
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
    runThreads(threadlist)
    print 'returned to main thread with', len(threadlist), 'threads'
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
    #check threadlist tos ee if everythong's there?
    if len(threadlist)==0:
        print 'Found nothing to update! Skipping position sizing!'
    else:
        print 'running vol_adjsize_live to update system files'
        with open(logPath+'vol_adjsize_live.txt', 'w') as f:
            with open(logPath+'vol_adjsize_live_error.txt', 'w') as e:
                proc = Popen(runPath2, stdout=f, stderr=e)
                proc.wait()
                
        print 'returned to main thread, running check systems if new orders are necessary.'
        with open(logPath+'check_systems_live.txt', 'w') as f:
            with open(logPath+'check_systems_live_error.txt', 'w') as e:
                proc = Popen(runPath4, stdout=f, stderr=e)
                proc.wait()
    totalc2orders=int(pd.read_sql('select * from checkSystems', con=conn).iloc[-1])
    
    #check ib positions
    try:
        execDict=update_orders(feeddata, systemfile, execDict)
    except Exception as e:
        #print e
        traceback.print_exc()
        
    iborders = [(sym, execDict[sym][:2]) for sym in execDict.keys() if execDict[sym][0] != 'PASS']
    num_iborders=len([execDict[sym][0] for sym in execDict.keys() if execDict[sym][0] != 'PASS'])
    
    if len(threadlist)==0 and totalc2orders ==0 and num_iborders==0:
        print 'Found nothing to update!'
    else:
        print 'Found', totalc2orders, 'c2 position adjustments.'
        print 'Found', num_iborders,'ib position adjustments.'
        #send orders if live mode
        if debug==False:
            print 'Live mode: running orders'
            if submitC2:
                for sys in systems:
                    print 'returned to main thread, running c2 orders for',sys
                    with open(logPath+'proc_signal_v4_live_'+sys+'.txt', 'a') as f:
                        with open(logPath+'proc_signal_v4_live_'+sys+'_error.txt', 'a') as e:
                            proc = Popen(runPath3+[sys], stdout=f, stderr=e)
                            proc.wait()
                            
                print 'returned to main thread, running check systems again..'
                with open(logPath+'check_systems_live.txt', 'a') as f:
                    with open(logPath+'check_systems_live_error.txt', 'a') as e:
                        proc = Popen(runPath4, stdout=f, stderr=e)
                        proc.wait()
            else:
                print 'submitC2 set to False'
            
            if submitIB:
                print 'returned to main thread, placing ib orders from', systemfile 
                try:
                    place_iborders(execDict)
                    executions=filterIBexec()
                    
                    #check executions
                    #drop data we don't need.
                    executions2 = executions.reset_index().groupby(['symbol','side'])[['qty']].max()

                    for (sym,[order,qty]) in iborders:
                        if (sym,order) in executions2.index and executions2.ix[sym].qty[0] ==qty:
                            execDict[sym][0] = 'PASS'
                            execDict[sym][1] = 0
                        else:
                            print 'There was an error:',sym,'order',  execDict[sym][:2], 'ib returned',\
                                executions2.ix[sym].index[0], executions2.ix[sym].qty[0]
                except Exception as e:
                    #print e
                    traceback.print_exc()
                num_iborders=len([execDict[sym][0] for sym in execDict.keys() if execDict[sym][0] != 'PASS'])
                print 'Found', num_iborders,'ib position adjustments after placing orders.'
            else:
                print 'submitIB set to False'
             
            totalc2orders=int(pd.read_sql('select * from checkSystems', con=conn).iloc[-1])
            print 'Found', totalc2orders, 'c2 position adjustments.'
        else:
            print 'Debug mode: skipping orders'

                
    #update slippage report
    
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
    
    
    #symbols = execDict.keys()
    #contracts = [x[2] for x in execDict.values()]
    '''
    def onExit():
        #run voladjsize_live
        print 'DONE!!!!!'
    sym='EMD'
    popenArgs = ['python', runPath,sym,'0']
    popenArgs = ['python', runPath]
    popenArgs2 = ['python', runPath2, sym]
    popenArgs2 = ['python', runPath2]
    '''
    #place_iborders(execDict)
    #contractsDF = get_contractdf(execDict, systemPath)
    #timetable= get_timetable(execDict, systemPath)

    #client.get_realtimebar(contracts[0], tickerId, whatToShow, data, filename)
    #client.get_history(endDateTime, execDict['LE'][2], whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
