import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import time
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import sqlite3

start_time = time.time()

if len(sys.argv)==1:
    debug=True
    showPlots=True
    commission=1
    start_slip=20161111
    figsize=(6,8)
    fontsize=12
    dataPath='D:/ML-TSDP/data/'
    portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
    #savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    systemPath =  'D:/ML-TSDP/data/systems/'
    #timetablePath=   'D:/ML-TSDP/data/systems/timetables_debug/'
    timetablePath=   'D:/ML-TSDP/data/systems/timetables/'
    feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
else:
    debug=False
    showPlots=False
    commission=1
    start_slip=20161111
    figsize=(8,13)
    fontsize=20
    dbPath='./data/futures.sqlite3'
    dataPath=savePath='./data/'
    portfolioPath = savePath2 ='./data/portfolio/'
    pngPath = './data/results/'
    systemPath =  './data/systems/'
    timetablePath=   './data/systems/timetables/'
    feedfile='./data/systems/system_ibfeed.csv'


conn = sqlite3.connect(dbPath)
atrFilename = 'futuresATR.csv'
systems = ['v4futures','v4mini', 'v4micro']
#systems = ['v4micro']
#systems = ['v4futures']

#adjustments
adjDict={
            '@CT':100,
            '@JY':0.01,
            'QSI':0.01
            }
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        
def plotSlip(slipDF, pngPath, filename, title, figsize, fontsize, showPlots=False):
    #plt.figure(figsize=(8,13))
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=figsize) # Create matplotlib figure
    ax = fig.add_subplot(111) # Create matplotlib axes
    #ax = slipDF.slippage.plot.bar(color='r', width=0.5)
    ax2 = ax.twiny()

    width=.3
    slipDF.slippage.plot(kind='barh', color='red',width=width, ax=ax, position=1)
    slipDF.delta.plot(kind='barh', color='blue', width=width,ax=ax2, position=0)

    ax.set_xlabel('Slippage % (red)')
    ax2.set_xlabel('Slippage Minutes (blue)')
    ax.grid(b=True)
    ax2.grid(b=False)
    #ax2 = slipDF.hourdelta.plot.bar(color='b', width=0.5)
    #plt.axvline(0, color='k')
    plt.text(0.5, 1.08, title,
             horizontalalignment='center',
             fontsize=fontsize,
             transform = ax2.transAxes)
    #plt.ylim(0,80)
    #plt.xticks(np.arange(-1,1.25,.25))
    #plt.grid(True)

    if pngPath != None and filename != None:
        plt.savefig(pngPath+filename, bbox_inches='tight')
        print 'Saved '+pngPath+filename
    if len(sys.argv)==1 and showPlots:
        #print data.index[0],'to',data.index[-1]
        plt.show()
    plt.close()
    
futuresDF = pd.read_csv(dataPath+atrFilename, index_col=0)
csidate = futuresDF.index.name.split()[0].replace('-','')
feeddata=pd.read_csv(feedfile,index_col='c2sym')

#import timetable
#ttfiles = os.listdir(timetablePath)
#ttdates = []
#for f in ttfiles:
#    if '.csv' in f and is_int(f.split('.')[0]):
#        ttdates.append(int(f.split('.')[0]))
#ttdate=max(ttdates)
if os.path.isfile(timetablePath+str(csidate)+'.csv'):
    timetable = pd.read_csv(timetablePath+str(csidate)+'.csv', index_col=0)
    for col in timetable.columns:
        timetable[col]=pd.to_datetime(timetable[col])
    idx_close = [x for x in timetable.index if 'close' in x]
    idx_trigger = [x for x in timetable.index if 'trigger' in x]
else:
    print 'timetable for', csidate,'not found! Ending script.'
    systems=[]


for systemName in systems:
    print systemName
    tradeFilename='c2_'+systemName+'_trades.csv'
    portfolioFilename = 'c2_'+systemName+'_portfolio.csv'
    systemFilename='system_'+systemName+'.csv'
    system = pd.read_csv(systemPath+systemFilename)
    
    
    #entry trades
    portfolioDF = pd.read_csv(portfolioPath+portfolioFilename)
    portfolioDF.index = feeddata.ix[[x[:-2] for x in portfolioDF.symbol.values]].ibsym.values
    portfolioDF = portfolioDF.ix[[True if type(x)==str else False for x in portfolioDF.index]]
    portfolioDF['openedWhen'] = pd.to_datetime(portfolioDF['openedWhen'])
    portfolioDF = portfolioDF.sort_values(by='openedWhen', ascending=False)
    #exit trades
    tradesDF = pd.read_csv(portfolioPath+tradeFilename)
    tradesDF.index = feeddata.ix[[x[:-2] for x in tradesDF.symbol.values]].ibsym.values
    tradesDF=tradesDF.ix[[True if type(x)==str else False for x in tradesDF.index]]
    tradesDF=tradesDF.drop(['expir','putcall','strike','symbol_description','underlying','markToMarket_time'], axis=1).dropna()
    tradesDF['closedWhen'] = pd.to_datetime(tradesDF['closedWhen'])
    tradesDF = tradesDF.sort_values(by='closedWhen', ascending=False)
    #csi close data
    
    #csidata download at 8pm est
    #futuresDate = dt.strptime(futuresDF.index.name, '%Y-%m-%d %H:%M:%S').replace(hour=20)
    slipDF = pd.DataFrame()
    #print 'sym', 'c2price', 'csiPrice', 'slippage'
    #new entry trades
    #newOpen=portfolioDF[portfolioDF['openedWhen']>=futuresDate].symbol.values
    newOpen={}
    system.index= system.c2sym
    for x in idx_trigger:
        sym=x.split()[0]
        if sym in portfolioDF.index:
            if portfolioDF.index.tolist().count(sym) == 1:
                row=portfolioDF.ix[sym]
            else:
                row=portfolioDF.ix[sym].iloc[0]
            if csidate in timetable:
                print sym,'trigger', timetable.ix[x][csidate],
                if row.openedWhen>=timetable.ix[x][csidate]:
                    print 'new open', row.openedWhen,'adding..'
                    newOpen[row.symbol]=[timetable.ix[x][csidate], row.quant_opened]
                else:
                    print ' no open trades found skipping..'
            else:
                print x,csidate,'not found in timetable, skipping..'
        else:
            print sym,' no open trades found skipping..'
            
    print systemName, len(newOpen.keys()), 'Open Trades Found'
    print systemName, newOpen.keys()
    for contract in newOpen.keys():
        c2price=portfolioDF[portfolioDF.symbol ==contract].opening_price_VWAP.values[0]
        c2timestamp=pd.Timestamp(portfolioDF[portfolioDF.symbol ==contract].openedWhen.values[0])
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            signal = system.ix[contract].signal
            qty =newOpen[contract][1]
            commissions = commission*qty
            cv = futuresDF[futuresDF.Contract ==contract].contractValue.values[0]
            dollarslip = int(-slippage*signal*qty*cv)
            #print contract, c2price, csiPrice,slippage
            #rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            rowName = 'ctwo: '+str(c2timestamp)+' '+str(qty)+' '+contract+' $'+str(dollarslip)
            slipDF.set_value(rowName, 'contract', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'trigger', newOpen[contract][0])
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Open')
            slipDF.set_value(rowName, 'signal', signal)
            slipDF.set_value(rowName, 'dollarslip', dollarslip)
            slipDF.set_value(rowName, 'commissions', commissions)
    #newCloses=tradesDF[tradesDF['closedWhen']>=futuresDate]
    newCloses=pd.DataFrame()
    
    for x in idx_trigger:
        sym=x.split()[0]
        if sym in tradesDF.index:
            if tradesDF.index.tolist().count(sym) == 1:
                row=tradesDF.ix[sym]
            else:
                row=tradesDF.ix[sym].iloc[0]

            if csidate in timetable:
                print sym,'trigger', timetable.ix[x][csidate],
                if row.closedWhen>=timetable.ix[x][csidate]:
                    print 'new close', row.closedWhen,'adding..'
                    row['trigger']=timetable.ix[x][csidate]
                    newCloses = newCloses.append(row)
                else:
                    print 'closed before trigger skipping..'
            else:
                print x,csidate,'not found in timetable, skipping..'
        else:
            print sym,' no close trades found skipping..'
            
    print systemName, newCloses.shape[0], 'Close Trades Found'
    newCloses.index = newCloses.symbol
    print systemName, newCloses.index
    
    for contract in newCloses.index:
        c2price=newCloses.ix[contract].closing_price_VWAP
        c2timestamp=pd.Timestamp(newCloses.ix[contract].closedWhen)
        trigger = pd.Timestamp(newCloses.ix[contract].trigger)
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            if newCloses.ix[contract].long_or_short == 'long':
                signal=-1
            else:
                signal =1
            qty = newCloses.ix[contract].quant_closed
            commissions = commission*qty
            cv = futuresDF[futuresDF.Contract ==contract].contractValue.values[0]
            dollarslip = int(-slippage*signal*qty*cv)
            #print contract, c2price, csiPrice,slippage
            #rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            rowName = 'ctwo: '+str(c2timestamp)+' '+str(qty)+' '+contract+' $'+str(dollarslip)
            slipDF.set_value(rowName, 'contract', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'trigger', trigger)
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Close')
            slipDF.set_value(rowName, 'signal', signal)
            slipDF.set_value(rowName, 'dollarslip', dollarslip)
            slipDF.set_value(rowName, 'commissions', commissions)
            
    if slipDF.shape[0]==0:
        print systemName, 'No new trades yesterday, skipping daily report'
    else:
        slipDF['timedelta']=slipDF.c2timestamp-slipDF.trigger
        slipDF['delta']=slipDF['timedelta'].astype('timedelta64[m]')
        #slipDF['delta']=slipDF.timedelta/np.timedelta64(1,'D')
        #if slipDF.shape[0] != portfolioDF.shape[0]:
        #    print 'Warning! Some values may be mising'

        
        openedTrades = slipDF[slipDF['Type']=='Open'].sort_values(by='abs_slippage', ascending=True)
        totalslip = int(openedTrades.dollarslip.sum())
        filename=systemName+'_open_slippage.png'
        title = systemName+': '+str(openedTrades.shape[0])+' Opened Trades, $'+str(totalslip)\
                    +' Slippage, CSI Data as of '+str(csidate)
        if openedTrades.shape[0] !=0:
            plotSlip(openedTrades, pngPath, filename, title, figsize, fontsize, showPlots=showPlots)
        else:
            print title
            
        closedTrades = slipDF[slipDF['Type']=='Close'].sort_values(by='abs_slippage', ascending=True)
        totalslip = int(closedTrades.dollarslip.sum())
        title = systemName+': '+str(closedTrades.shape[0])+' Closed Trades, $'+str(totalslip)\
                    +' Slippage, CSI Data as of '+str(csidate)
        filename=systemName+'_close_slippage.png'
        if closedTrades.shape[0] != 0:
            plotSlip(closedTrades, pngPath, filename, title, figsize, fontsize, showPlots=showPlots)
        else:
            print title
            
        slipDF.index.name = 'rowname'
        filename=systemName+'_slippage_report_'+str(csidate).split()[0].replace('-','')+'.csv'
        slipDF = slipDF.sort_values(by='abs_slippage', ascending=True)
        slipDF.to_csv(savePath+systemName+'_slippage_report.csv', index=True)
        print 'Saved '+savePath+systemName+'_slippage_report.csv'
        slipDF.to_csv(savePath2+filename, index=True)
        print 'Saved '+savePath2+filename
        slipDF['Name']=systemName
        slipDF['ibsym']=feeddata.ix[[x[:-2] for x in slipDF.contract.values]].ibsym.values
        slipDF['csiDate']=csidate
        slipDF['timestamp']=int(time.mktime(dt.utcnow().timetuple()))
        slipDF.to_sql(name= 'slippage', if_exists='append', con=conn, index=True, index_label='CSIsym')
        print 'Saved '+systemName+' to sql'

    ###########################################################
    #average slippage file/png
    files=os.listdir(portfolioPath)
    slipFiles = [x for x in files if systemName+'_slippage_report_' in x]
    slipFiles = [x for x in slipFiles if is_int(x.split('_')[-1].split('.')[0])]
    slipFiles = [x for x in slipFiles if int(x.split('_')[-1].split('.')[0])>start_slip]
    
    cons = pd.DataFrame()
    for f in slipFiles:
        fi = pd.read_csv(portfolioPath+f,index_col='rowname')
        cons=cons.append(fi)
    if cons.shape[0] !=0:
        avgslip=pd.DataFrame()
        for sym in cons.contract.unique():
            trades =len(cons[cons.contract==sym].c2timestamp.unique())
            abs_slip=abs(cons[cons.contract==sym].abs_slippage.mean())
            delta=cons[cons.contract==sym].delta.mean()
            #print sym, abs_slip
            avgslip.set_value(sym, 'slippage', abs_slip)
            avgslip.set_value(sym, 'delta', delta)
            avgslip.set_value(sym, 'trades', trades)
        avgslip=avgslip.sort_values(by='slippage', ascending=True)
        print systemName, str(avgslip.shape[0]), 'Symbols found in the average absolute slippage DF...'
        print avgslip.index.values
        '''
        i=0
        for x in futuresDF[futuresDF.finalQTY != 0].Contract:
            if x not in cons.symbol.unique():
                i+=1
                print x,
        print i, 'Symbols missing!'
        '''
        index = str(trades)+'trades '+sym
        
        system_sym=[ x for x  in system[system.c2qty !=0].c2sym.values if x in avgslip.index.values]
        system_slip=avgslip.ix[system_sym].sort_values(by='slippage', ascending=True)
        system_slip.index = [str(int(system_slip.ix[i].trades))+' trades '+i for i in system_slip.index]
        filename=systemName+'_avg_slippage.png'
        if len(slipFiles)>1:
            title=systemName+' Avg. Slippage of '+str(system_slip.shape[0])+' Contracts from '\
                    +slipFiles[1].split('_')[3][:-4]+' to '+slipFiles[-1].split('_')[3][:-4]
        else:
            title=systemName+' Avg. Slippage of '+str(system_slip.shape[0])+' Contracts from '\
                    +slipFiles[-1].split('_')[3][:-4]
        plotSlip(system_slip, pngPath, filename, title,figsize, fontsize, showPlots=showPlots)
        #slippage by system


        filename=systemName+'_slip_cons.csv'
        cons.to_csv(savePath2+filename, index=True)
        print 'Saved '+savePath2+filename+'\n'
    else:
        print 'no slippage reports found. skipping consolidated report.'

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()