import socket   
import select
import sys
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import subprocess
import json
import time
from pandas.io.json import json_normalize
import pandas as pd
import threading
#from btapi.get_signal import get_v1signal
#from btapi.get_hist_btcharts import get_bthist
#from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv
#from seitoolz.paper import adj_size
from suztoolz.debug_system_v2_30min_func import runv2
import sys
#import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
#import websocket
from suztoolz.display import offlineMode
import seitoolz.bars as bars
from multiprocessing import Process, Queue
import os
from pytz import timezone
from dateutil.parser import parse
import sys
import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

    


def get_last_bars_debug(currencyPairs, ylabel, callback, **kwargs):
    global tickerId
    global lastDate
    minPath = kwargs.get('minPath','./data/bars/')
    #while 1:
    try:
        SST=pd.DataFrame()
        symbols=list()
        returnData=False
        for i,ticker in enumerate(currencyPairs):
            pair=ticker
            minFile=minPath+pair+'.csv'
            symbol = pair
            #print minFile
            if os.path.isfile(minFile):
                logging.info(str(i)+' loading '+minFile)
                dta=pd.read_csv(minFile).iloc[-1]
                date=dta['Date']
                
                eastern=timezone('US/Eastern')
                date=parse(date).replace(tzinfo=eastern)
                timestamp = time.mktime(date.timetuple())
                dta.Date = date
                #data=pd.DataFrame()
                #data['Date']=date
                #data[symbol]=dta[ylabel]
                #print dta[ylabel]
                #print data, date
                #data=data.set_index('Date') 
                dta.name = dta.Date
                
                #if len(SST.index.values) < 1:
                #    SST=dta
                #else:
                 #   SST=SST.join(dta)
                    
                if not lastDate.has_key(symbol):
                    lastDate[symbol]=timestamp
                                           
                if lastDate[symbol] < timestamp:
                    returnData=True
                symbols.append(symbol)
                        
            #if returnData:
            #data=SST
            #data=data.set_index('Date')
            #data=data.fillna(method='pad')
        callback(dta, symbols)
            #time.sleep(20)
    except Exception as e:
        logging.error("get_last_bar", exc_info=True)
        
def get_bars(pairs, interval):
    #global SST
    #global start_time
    mypairs=list()
    for pair in pairs:
        mypairs.append(interval + pair)
        
    if debug:
        get_last_bars_debug(mypairs, 'Close', onBar,minPath=minPath)
    else:
        bars.get_last_bars(mypairs, 'Close', onBar)
        #get_last_bars_debug(mypairs, 'Close', onBar)

def onBar(bar, symbols):
    global start_time
    global gotbar
    global pairs
    bar = bar.iloc[-1]
    logging.info('received '+str(symbols)+str(bar))
    if not gotbar.has_key(bar['Date']):
        gotbar[bar['Date']]=list()
    #print bar['Date'], gotbar[bar['Date']]
    for symbol in symbols:
        if symbol not in gotbar[bar['Date']]:
            gotbar[bar['Date']].append(symbol)
    #gotbar[bar['Date']]=[i for sublist in gotbar[bar['Date']] for i in sublist]
    logging.info(str(bar['Date'])+ str(gotbar[bar['Date']])+ str(len(gotbar[bar['Date']]))+'bars '+ str(len(pairs))+'pairs')
    #global SST
    #SST = SST.combine_first(bar).sort_index()
    #if debug:
    
    if len(gotbar[bar['Date']])==len(livePairs):
    #if len([p for p in gotbar[bar['Date']] if p in livePairs]) == len(livePairs):
        #print gotbar[bar['Date']]
        start_time2 = time.time()
        #for sym in gotbar[bar['Date']]:
        logging.info('')
        logging.info('timenow: '+dt.now(timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S %Z"))
        runPairs()
        logging.info( 'All signals created for bar '+str(bar['Date']))
        logging.info('Runtime: '+str(round(((time.time() - start_time2)/60),2))+ ' minutes' ) 
        logging.info('Last bar time: '+str(round(((time.time() - start_time)/60),2))+ ' minutes\n' ) 
        start_time = time.time()
    #else:   
    #    if len(gotbar[bar['Date']])==len(pairs):
    #        print  len(gotbar[bar['Date']]), 'bars collected for', bar['Date'],'running systems..'
     #       for sym in gotbar[bar['Date']]:
                #print sym
     #           runPair_v1(sym)
                
    
def runPairs():

    debug=False
    savePath = './data/results/'
    dataPath = './data/from_IB/'
    signalPath = './data/signals/'
    pairPath='./data/'

    with open(pairPath+'currencies.txt') as f:
        currencyPairs = f.read().splitlines()
    version='v4'
    verbose=True
    lookback=1
    currencies = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD']
    #currencies = ['EUR', 'GBP', 'JPY', 'USD']
    barSizeSetting='30m'
    for i in range(7,-1,-lookback):
        #startDate=dt(2016, 5, i,0,00)
        cMatrix=pd.DataFrame()
        for currency in currencies:
            #lookback=i
            #cMatrix[currency]=pd.DataFrame()
            pairList=[pair for pair in currencyPairs if currency in pair[0:3] or currency in pair[3:6]]
            pairList =[pair for pair in pairList if pair[0:3] in currencies and pair[3:6] in currencies]
            #files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
            for pair in pairList:    
                #logging.info( -(i+1), -(1-lookback+1)
                if i ==0:
                    data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)[-(i+2):]
                else:
                    data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)[-(i+2):-(i-lookback+1)]
                #data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)
                data.index = data.index.to_datetime()
                #lookback=data.ix[startDate:].shape[0]
                if currency in pair[0:3]:
                    #logging.info( pair[3:6],currency,data.Close.pct_change(periods=lookback)[-1]*100
                    #cMatrix.set_value(pair[3:6],currency,-data.Close.pct_change(periods=lookback)[-1]*100)
                    cMatrix.set_value(pair[3:6],currency,-data.Close.pct_change(periods=lookback)[-1]*100)
                else:
                    #logging.info( pair[0:3],currency,-data.Close.pct_change(periods=lookback)[-1]*100
                    #pair2=pair[3:6]+pair[0:3]
                    #cMatrix.set_value(pair[0:3], currency,data.Close.pct_change(periods=lookback)[-1]*100)
                    cMatrix.set_value(pair[0:3], currency,data.Close.pct_change(periods=lookback)[-1]*100)
                    
        for currency in currencies:
            cMatrix.set_value(currency,'Avg',cMatrix.ix[currency].dropna().mean())
        cMatrix=cMatrix.sort_values(by='Avg', ascending=False).fillna(0)
        #cMatrix=cMatrix.fillna(0)
        rankByMean=cMatrix['Avg']
        '''
        with open(savePath+'currencies_1.html','w') as f:
            f.write(str(startDate)+' to '+str(data.index[-1]))
            
        cMatrix.to_html(savePath+'currencies_4.html')
        '''
        #logging.info( data.index[0],'to',data.index[-1]
        #logging.info( cMatrix
        fig,ax = plt.subplots(figsize=(8,8))
        sns.heatmap(ax=ax,data=cMatrix)
        #ax.set_title(str(data.ix[startDate].name)+' to '+str(data.index[-1]))
        startDate=data.index[0]
        ax.set_title(str(data.ix[startDate].name)+' to '+str(data.index[-1]))
        #plt.pcolor(cMatrix)
        #plt.yticks(np.arange(0.5, len(cMatrix.index), 1), cMatrix.index)
        #plt.xticks(np.arange(0.5, len(cMatrix.columns), 1), cMatrix.columns)
        if savePath != None:
            logging.info( 'Saving '+savePath+'currencies_'+str(i+2)+'.png')
            fig.savefig(savePath+'currencies_'+str(i+2)+'.png', bbox_inches='tight')
            
        if debug:
            #logging.info( startDate,'to',data.index[-1]
            plt.show()


        ranking = rankByMean.index
        buyHold=[]
        sellHold=[]

        cplist = copy.deepcopy(currencyPairs)
        for currency in ranking:
            for i,pair in enumerate(cplist):
                #logging.info( pair
                if pair not in buyHold and pair not in sellHold:
                    if currency in pair[0:3]:
                        #logging.info( i,'bh',pair
                        buyHold.append(pair)
                        #cplist.remove(pair)
                    elif currency in pair[3:6]:
                        #logging.info( i,'sh',pair
                        sellHold.append(pair)
                        #cplist.remove(pair)
                    #else:
                        #logging.info( i,currency,pair
        offline=[pair for pair in currencyPairs if pair not in buyHold+sellHold]
    if verbose:
        logging.info( str(startDate)+' to '+str(data.index[-1]))
        logging.info( 'Overall Rank\n'+str(rankByMean))
        logging.info( 'buyHold ',str(len(buyHold)),str(buyHold))
        logging.info( 'sellHold ',str(len(sellHold)),str(sellHold))
        logging.info( 'offline '+str(len(offline)),str(offline))
        
    nsig=0
    for ticker in buyHold:
        nsig+=1
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        #addLine = signalFile.iloc[-1]
        #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        #addLine.name = sst.iloc[-1].name
        addLine = pd.Series(name=data.index[-1])
        addLine['signals']=1
        addLine['safef']=1
        addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
        signalFile = signalFile.append(addLine)
        filename=signalPath + version+'_'+ ticker+ '.csv'
        #logging.info( 'Saving', filename
        signalFile.to_csv(filename, index=True)
        
    for ticker in sellHold:
        nsig+=1
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        #addLine = signalFile.iloc[-1]
        #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        #addLine.name = sst.iloc[-1].name
        addLine = pd.Series(name=data.index[-1])
        addLine['signals']=-1
        addLine['safef']=1
        addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
        signalFile = signalFile.append(addLine)
        filename=signalPath + version+'_'+ ticker+ '.csv'
        #logging.info( 'Saving', filename
        signalFile.to_csv(filename, index=True)
        
    for ticker in offline:
        nsig+=1
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        #addLine = signalFile.iloc[-1]
        #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        #addLine.name = sst.iloc[-1].name
        addLine = pd.Series(name=data.index[-1])
        addLine['signals']=0
        addLine['safef']=0
        addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
        signalFile = signalFile.append(addLine)
        filename=signalPath + version+'_'+ ticker+ '.csv'
        #logging.info( 'Saving', filename
        signalFile.to_csv(filename, index=True)
    logging.info( str(nsig)+'signals saved')
    
def runPair_v1(pair):
    offline=True
    ticker = pair[0].split('_')[1]
    version = 'v1'
    version_ = 'v1.3C'
    runDPS = False
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData,\
                'modelPath':modelPath,'loadModel':loadModel}
                
    if offline:
        try:
            with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                message = "Offline Mode: turned off in runsystem"
                offlineMode(ticker, message, signalPath, version, version_)
                logging.info(version+' '+ticker+' '+message)
            sys.stdout = orig_stdout
            runPair_v2(pair, dataSet)
        except Exception as e:
                 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
                 #ferr.write(e)
                 #ferr.close()
                 logging.error("something bad happened", exc_info=True)
    else:
        try:
            with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                print 'Starting '+version+': ' + ticker
                if ticker not in livePairs:
                    offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
                #f.write('Starting '+version+': ' + ticker)         
                #ferr.write('Starting '+version+': ' + ticker)
                signal, dataSet=runv2(runData)
                print signal
                logging.info('v1 '+' signal '+str(signal.signals)+ ' safef '+str(signal.safef)+' CAR25 '+str(signal.CAR25))
                logging.info(signal.system)
                #subprocess.call(['python','debug_system_v1.3C_30min.py',ticker,'1'], stdout=f, stderr=ferr)
                #f.close()
                #ferr.close()

            sys.stdout = orig_stdout
            runPair_v2(pair, dataSet)
        except Exception as e:
                 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
                 #ferr.write(e)
                 #ferr.close()
                 logging.error("something bad happened", exc_info=True)
 
def runPair_v2(pair, dataSet):
    offline=True
    ticker = pair[0].split('_')[1]
    version = 'v2'
    version_ = 'v2.4C'
    runDPS = True
    runData = {'ticker':ticker, 'showDist':showDist,'showPDFCDF':showPDFCDF,'showAllCharts':showAllCharts,\
                'runDPS':runDPS,'saveParams':saveParams,'saveDataSet':saveDataSet,'verbose':verbose,\
                'scorePath' : scorePath, 'equityStatsSavePath' : equityStatsSavePath,'signalPath' : signalPath,\
                'dataPath' :dataPath, 'bestParamsPath' :  bestParamsPath, 'chartSavePath' :chartSavePath,\
                'version':version, 'version_':version_, 'filterName':filterName, 'data_type':data_type,\
                'barSizeSetting':barSizeSetting,'currencyPairs':pairs, 'perturbData':perturbData,\
                'modelPath':modelPath,'loadModel':loadModel}
                
    if offline:
        try:
            with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                message = "Offline Mode: turned off in runsystem"
                offlineMode(ticker, message, signalPath, version, version_)
                logging.info(version+' '+ticker+' '+message)
            sys.stdout = orig_stdout
        except Exception as e:
                 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
                 #ferr.write(e)
                 #ferr.close()
                 logging.error("something bad happened", exc_info=True)
    else:
        try:
            with open ('/logs/' + version+'_'+ticker + 'onBar.log','a') as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                print 'Starting '+version+': ' + ticker
                #if ticker not in livePairs:
                #    offlineMode(ticker, "Offline Mode: turned off in runsystem", signalPath, version, version_)
                #f.write('Starting '+version+': ' + ticker)
                
                #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
                #ferr.write('Starting '+version+': ' + ticker)
                if dataSet is not None:
                    signal, dataSet=runv2(runData, dataSet)
                else:
                    signal, dataSet=runv2(runData)
                print signal
                logging.info('v2 '+' signal '+str(signal.signals)+ ' safef '+str(signal.safef)+' CAR25 '+str(signal.CAR25))
                logging.info(signal.system)
                #subprocess.call(['python','debug_system_v1.3C_30min.py',ticker,'1'], stdout=f, stderr=ferr)
                #f.close()
                #ferr.close()
            sys.stdout = orig_stdout
        except Exception as e:
                 #ferr=open ('/logs/' + version+'_'+ticker + 'onBar_err.log','a')
                 #ferr.write(e)
                 #ferr.close()
                 logging.error("something bad happened", exc_info=True)
             
def runThreads():
    global start_time
    start_time = time.time()
    threads = []
    for pair in pairs:
        sig_thread = threading.Thread(target=get_bars, args=[[pair], barSizeSetting+'_'])
        sig_thread.daemon=True
        threads.append(sig_thread)
        sig_thread.start()
        
    if debug==False:
        while 1:
            time.sleep(100)



logging.basicConfig(filename='/logs/runsystem_v4ConBar.log',level=logging.DEBUG)
start_time = time.time()
lastDate={}
tickerId=1
gotbar=dict()

#filterName = 'DF1'
#data_type = 'ALL'
barSizeSetting='30m'
pairPath='./data/'

with open(pairPath+'currencies.txt') as f:
    pairs =livePairs = f.read().splitlines()
        
                 
if len(sys.argv)==1:
    '''
    livePairs = [
                'NZDJPY',\
                'CADJPY',\
                'CHFJPY',\
                'EURJPY',\
                'GBPJPY',\
                'AUDJPY',\
                'USDJPY',\
                'AUDUSD',\
                'EURUSD',\
                'EURAUD',\
                'EURCAD',\
                'EURNZD',\
                'GBPUSD',\
                'USDCAD',\
                'USDCHF',\
                'NZDUSD',
                'EURCHF',\
                'EURGBP',\
                'AUDCAD',\
                'AUDCHF',\
                'AUDNZD',\
                'GBPAUD',\
                'GBPCAD',\
                'GBPNZD',\
                'GBPCHF',\
                'CADCHF',\
                'NZDCHF',\
                'NZDCAD'
                ]
    '''
    #settings
    debug=True
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    #runDPS = True
    saveParams = False
    saveDataSet=True
    verbose= False
    loadModel=True
    #paths
    scorePath = None
    equityStatsSavePath = None
    #scorePath = 'C:/users/hidemi/desktop/Python/scored_metrics_'
    #equityStatsSavePath = 'C:/Users/Hidemi/Desktop/Python/'
    signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    dataPath = 'D:/ML-TSDP/data/from_IB/'
    modelPath = 'D:/ML-TSDP/data/models/'
    bestParamsPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/params/' 
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/' 
    minPath= 'D:/ML-TSDP/data/bars/'
    #while 1:          
    get_bars(livePairs,barSizeSetting+'_')
    #time.sleep(100)
    
    
else:
    '''
    livePairs = [
                'NZDJPY',\
                'CADJPY',\
                'CHFJPY',\
                'EURJPY',\
                'GBPJPY',\
                'AUDJPY',\
                'USDJPY',\
                'AUDUSD',\
                'EURUSD',\
                'EURAUD',\
                'EURCAD',\
                'EURNZD',\
                'GBPUSD',\
                'USDCAD',\
                'USDCHF',\
                'NZDUSD',
                'EURCHF',\
                'EURGBP',\
                'AUDCAD',\
                'AUDCHF',\
                'AUDNZD',\
                'GBPAUD',\
                'GBPCAD',\
                'GBPNZD',\
                'GBPCHF',\
                'CADCHF',\
                'NZDCHF',\
                'NZDCAD'
                ]
    '''
    #settings
    debug=False
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    #runDPS = True
    saveParams = False
    saveDataSet=True
    verbose= False
    loadModel=False
    #paths
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
    modelPath = './data/models/'
    bestParamsPath =  './data/params/'
    chartSavePath = './data/results/' 
    minPath= './data/bars/'
    
    if len(sys.argv) >2:
        if sys.argv[2] == 'debug':  
            debug = True
            logging.info( 'running debug mode...' )
        
    if sys.argv[1] == 'single':  
        while 1:
            start_time = time.time()
            logging.info( 'starting single thread mode for '+str(barSizeSetting)+' bars '+str(len(livePairs))+' pairs.')
            logging.info(str(livePairs) )
            get_bars(livePairs,barSizeSetting+'_')
            #time.sleep(100)
    elif sys.argv[1] == 'multi':
        runThreads()
    else:
        #print 'please specify single or multi, thanks.'
        sys.exit('please specify single or multi.')
    

