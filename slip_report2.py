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

start_time = time.time()

if len(sys.argv)==1:
    debug=True
    showPlots=True
    dataPath='D:/ML-TSDP/data/'
    portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    #savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    systemPath =  'D:/ML-TSDP/data/systems/'
else:
    debug=False
    showPlots=False
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    savePath='./data/'
    pngPath = './data/results/'
    savePath2 = './data/portfolio/'
    systemPath =  './data/systems/'



csidataFilename = 'futuresATR_Signals.csv'
systems = ['v4futures','v4mini', 'v4micro']


#adjustments
adjDict={
            '@CT':100,
            '@JY':0.01,
            'QSI':0.01
            }

def plotSlip(slipDF, pngPath, filename, title, showPlots=False):
    #plt.figure(figsize=(8,13))
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,13)) # Create matplotlib figure
    ax = fig.add_subplot(111) # Create matplotlib axes
    #ax = slipDF.slippage.plot.bar(color='r', width=0.5)
    ax2 = ax.twiny()

    width=.3
    slipDF.slippage.plot(kind='barh', color='red',width=width, ax=ax, position=1)
    slipDF.delta.plot(kind='barh', color='blue', width=width,ax=ax2, position=0)

    ax.set_xlabel('Slippage % (red)')
    ax2.set_xlabel('Slippage Days (blue)')
    ax.grid(b=True)
    ax2.grid(b=False)
    #ax2 = slipDF.hourdelta.plot.bar(color='b', width=0.5)
    #plt.axvline(0, color='k')
    plt.text(0.5, 1.08, title,
             horizontalalignment='center',
             fontsize=20,
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


for systemName in systems:
    tradeFilename='c2_'+systemName+'_trades.csv'
    portfolioFilename = 'c2_'+systemName+'_portfolio.csv'
    systemFilename='system_'+systemName+'.csv'


    #entry trades
    portfolioDF = pd.read_csv(portfolioPath+portfolioFilename)
    portfolioDF['openedWhen'] = pd.to_datetime(portfolioDF['openedWhen'])
    portfolioDF = portfolioDF.sort_values(by='openedWhen', ascending=False)
    #exit trades
    tradesDF = pd.read_csv(portfolioPath+tradeFilename)
    tradesDF=tradesDF.drop(['expir','putcall','strike','symbol_description','underlying','markToMarket_time'], axis=1).dropna()
    tradesDF['closedWhen'] = pd.to_datetime(tradesDF['closedWhen'])
    tradesDF = tradesDF.sort_values(by='closedWhen', ascending=False)
    #csi close data
    futuresDF = pd.read_csv(dataPath+csidataFilename, index_col=0)
    #csidata download at 8pm est
    futuresDate = dt.strptime(futuresDF.index.name, '%Y-%m-%d %H:%M:%S').replace(hour=20)
    slipDF = pd.DataFrame()
    #print 'sym', 'c2price', 'csiPrice', 'slippage'
    #new entry trades
    newOpen=portfolioDF[portfolioDF['openedWhen']>=futuresDate].symbol.values
    for contract in newOpen:
        c2price=portfolioDF[portfolioDF.symbol ==contract].opening_price_VWAP.values[0]
        c2timestamp=pd.Timestamp(portfolioDF[portfolioDF.symbol ==contract].openedWhen.values[0])
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            #print contract, c2price, csiPrice,slippage
            rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            slipDF.set_value(rowName, 'symbol', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'csitimestamp', futuresDate)
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Open')

    newCloses=tradesDF[tradesDF['closedWhen']>=futuresDate]
    for contract in newCloses.symbol.values:
        c2price=newCloses[newCloses.symbol ==contract].closing_price_VWAP.values[0]
        c2timestamp=pd.Timestamp(newCloses[newCloses.symbol ==contract].closedWhen.values[0])
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            #print contract, c2price, csiPrice,slippage
            rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            slipDF.set_value(rowName, 'symbol', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'csitimestamp', futuresDate)
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Close')
            
    slipDF['timedelta']=slipDF.c2timestamp-slipDF.csitimestamp
    slipDF['delta']=slipDF.timedelta/np.timedelta64(1,'D')
    if slipDF.shape[0] != portfolioDF.shape[0]:
        print 'Warning! Some values are mising'

    
    openedTrades = slipDF[slipDF['Type']=='Open'].sort_values(by='abs_slippage', ascending=True)
    filename=systemName+'_open_slippage.png'
    title = systemName+' '+str(openedTrades.shape[0])+' Opened Trades, CSI Data as of '+str(futuresDate)
    if openedTrades.shape[0] !=0:
        plotSlip(openedTrades, pngPath, filename, title, showPlots=showPlots)
    else:
        print title
        
    closedTrades = slipDF[slipDF['Type']=='Close'].sort_values(by='abs_slippage', ascending=True)
    title = systemName+' '+str(closedTrades.shape[0])+' Closed Trades, CSI Data as of '+str(futuresDate)
    filename=systemName+'_close_slippage.png'
    if closedTrades.shape[0] != 0:
        plotSlip(closedTrades, pngPath, filename, title, showPlots=showPlots)
    else:
        print title
        
    slipDF.index.name = 'rowname'
    filename=systemName+'_slippage_report_'+str(futuresDate).split()[0].replace('-','')+'.csv'
    slipDF = slipDF.sort_values(by='abs_slippage', ascending=True)
    slipDF.to_csv(savePath+systemName+'_slippage_report.csv', index=True)
    print 'Saved '+savePath+systemName+'_slippage_report.csv'
    slipDF.to_csv(savePath2+filename, index=True)
    print 'Saved '+savePath2+filename

    #average slippage file/png
    files=os.listdir(savePath2)
    slipFiles = [x for x in files if systemName+'_slippage_report' in x]

    cons = pd.DataFrame()
    for f in slipFiles:
        fi = pd.read_csv(savePath2+f,index_col='rowname')
        cons=cons.append(fi)
        
    avgslip=pd.DataFrame()
    for sym in cons.symbol.unique():
        abs_slip=abs(cons[cons.symbol==sym].abs_slippage.mean())
        delta=cons[cons.symbol==sym].delta.mean()
        #print sym, abs_slip
        avgslip.set_value(sym, 'slippage', abs_slip)
        avgslip.set_value(sym, 'delta', delta)
    avgslip=avgslip.sort_values(by='slippage', ascending=True)
    print str(avgslip.shape[0]), 'Symbols found in the average absolute slippage DF...'
    
    i=0
    for x in futuresDF[futuresDF.finalQTY != 0].Contract:
        if x not in cons.symbol.unique():
            i+=1
            print x,
    print i, 'Symbols missing!'

    system = pd.read_csv(systemPath+systemFilename)
    system_sym=system[system.c2qty !=0].c2sym.values
    system_slip=avgslip.ix[system_sym].sort_values(by='slippage', ascending=True)
    filename=systemName+'_avg_slippage.png'
    title=systemName+' Avg. Slippage of '+str(system_slip.shape[0])+' Contracts from '\
            +slipFiles[1].split('_')[3][:-4]+' to '+slipFiles[-1].split('_')[3][:-4]
    plotSlip(system_slip, pngPath, filename, title, showPlots=showPlots)
    #slippage by system


    filename=systemName+'_slip_cons.csv'
    cons.to_csv(savePath2+filename, index=True)
    print 'Saved '+savePath2+filename+'\n'

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()